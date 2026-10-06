# GPU readiness: routing tensor-shaped operations through tenferro-rs

Status: design and inventory (issue #417, related to #185, #360, #361). No production
code changes in #417. The `gpu-cuda` build path and optional gate landed with issue #420
(section 10); the validation harness of #424 is described in `docs/NUMERICAL_COMPARISONS.md`
("Accelerated-backend validation"). Measurements were taken on `origin/main` at `dfa9515f`.

The goal is to let the tensor-shaped part of mvmc-rs move to a GPU backend later without
rewriting the physics code. This document inventories the operations, records what
tenferro-rs can and cannot do today, and fixes the architecture, validation policy and
roadmap. It does not change the numerical contract in `AGENTS.md` and
`docs/NUMERICAL_COMPARISONS.md`.

## 1. Summary

1. **The CPU C-order path is and stays the reference.** Accelerated paths are additional
   implementations behind a coarse, stage-level backend trait. They never replace the scalar
   path, and no CPU test depends on a GPU or on tenferro GPU crates.
2. **On the benchmark inputs the GPU opportunity is the Pfaffian/inverse, not SR.**
   For the Hubbard-chain inputs (L=16/32/64, one thread, direct SR) the batched
   Pfaffian+inverse (`CalculateMAll` plus the sampler's `recal PfM and InvM`) is 58-61 % of
   wall time, the sampler's rank-one updates are 16-30 %, local energy plus Slater
   derivative 6-10 %, and the SR S-matrix/force/solve are only 0.2-0.3 % (0.6-1.5 % with the
   O-vector store and Gram product). SR becomes dominant only when `NPara` is in the
   thousands (section 3.4).
3. **tenferro-rs 0.7.1 already provides what the SR/GEMM-shaped work needs**
   (`dot_general`/einsum with batch dims, F64/C64, `cholesky`, `triangular_solve`, `solve`,
   CUDA through cuBLAS/cuSOLVER/cuTENSOR, explicit upload/download). It does **not** provide
   a skew-symmetric factorization or Pfaffian, has no documented reduction-order or
   determinism contract, and has no asynchronous or pinned transfer. Those are the upstream
   requests in section 4.7.
4. **A single Markov chain cannot profit from a GPU.** One attempted hop costs 2-13 us on one
   CPU core; one device round trip costs tens of microseconds. The device is only useful with
   hundreds of independent walkers per device (section 5.5), each with its own host SFMT
   stream seeded `RndSeed + group`, exactly the C contract.
5. **Migration order.** First a low-risk SR/CG pathfinder that exercises the tenferro tensor
   boundary, then the batched Pfaffian/inverse (the real benefit), then walker batching, then
   the local-energy tables. Section 6 gives the phases and the Amdahl upper bounds (2x after
   the Pfaffian stage, 4-9x with the sampler, 8-19x with measurements, all before transfer and
   launch costs).

## 2. Notation and problem sizes

| symbol | meaning | source |
| --- | --- | --- |
| `Nsite` | lattice sites | `modpara.def` |
| `Ne` | electrons per spin (`Ncond/2` at `2Sz=0`) | |
| `Nsize` (`n`) | `2*Ne`, Pfaffian matrix side | `pfaffian.rs:134` doc |
| `NQP` | `NSPGaussLeg * NMPTrans * NQPOptTrans` (`n_qp_full`) | `slater_update.rs:17` |
| `NPara` | `NProj + NSlater (+ NOptTrans + RBM)`; SR size is `1 + NPara` | `state.rs:420` |
| `NVMCSample` (`NS`) | saved configurations per SR step | `modpara.def` |
| `B` | independent chains (walkers) per device (proposed) | section 5 |

Hubbard-chain benchmark inputs (`benchmark/hubbard_chain/inputs`, half filling, real mode):

| input | Nsite | Nsize | NQP | NProj | NSlater | NPara | NS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L16 | 16 | 16 | 8 | 4+32 | 64 | 100 | 300 |
| L32 | 32 | 32 | 8 | 4+64 | 128 | 196 | 300 |
| L64 | 64 | 64 | 8 | 4+128 | 256 | 388 | 300 |

`NQP = 8` is measured, not assumed: the inner profile reports 8 plane items per
`calc_m_all` call (`NSPGaussLeg=8`, `NMPTrans=-1` resolves to one momentum projection,
`NQPOptTrans=1`). Matrices are small (`n` = 16-64) and the per-chain QP batch is only 8.
That is why batching must also span samples and walkers.

## 3. Inventory

### 3.1 Method

`MVMC_C_TIMER=1 MVMC_RS_INNER_PROFILE=1`, release CLI built from `dfa9515f`
(`cargo build --release -p mvmc-cli`), OpenBLAS (`libopenblas.so.0`, system), 1 thread
(`OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 MVMC_RS_INNER_THREADS=1`), Intel Xeon E5-2699 v3
@ 2.30 GHz, Linux 6.8, rustc 1.99.0, `NVMCSample=300`, `--nsmp 10`, 20 SR steps (L16, L32) and
10 steps (L64, to bound runtime). Single run per size on a shared host; section timers are
inclusive and nested sections overlap, so percentages of `All` are the robust signal, not the
absolute seconds. Reproduce with:

```sh
cd benchmark/hubbard_chain/inputs/hubbard_chain_L32   # or L16, L64 on a copy
OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 MVMC_RS_INNER_THREADS=1 \
  MVMC_C_TIMER=1 MVMC_RS_INNER_PROFILE=1 \
  mvmc --nsteps 20 --nsmp 10 --out-dir /tmp/out namelist.def 2> profile.txt
# section table: /tmp/out/zvo_CalcTimer.dat; per-call-site table: profile.txt (stderr)
```

PhysCal (`--physcal`) shares were not re-measured here. They are taken from the committed memo
`benchmark/physcal/results/physcal_hubbard_2026-10-04_sections.md` (older commit, L32,
`NDataQtySmp=100`): `CalculateMAll` 26 %, `LocEnergyCal` 12 %, `CalculateGreenFunc` 7.6 % of `All`.

### 3.2 Section shares (seconds, percent of `All`)

| section | L=16 | L=32 | L=64 |
| --- | ---: | ---: | ---: |
| `All [0]` | 0.824 (100) | 3.197 (100) | 7.574 (100) |
| `VMCMakeSample [3]` | 0.340 (41.3) | 1.375 (43.0) | 3.748 (49.5) |
| `  hopping update [32]` | 0.191 (23.1) | 0.820 (25.7) | 2.507 (33.1) |
| `    UpdateMAll [63]` | 0.105 (12.8) | 0.569 (17.8) | 2.080 (27.5) |
| `    CalculateNewPfM2 [61]` | 0.033 (4.1) | 0.101 (3.2) | 0.177 (2.3) |
| `    UpdateProjCnt [60]` | 0.016 (1.9) | 0.050 (1.6) | 0.109 (1.4) |
| `  recal PfM and InvM [34]` | 0.128 (15.5) | 0.508 (15.9) | 1.179 (15.6) |
| `VMCMainCal [4]` | 0.463 (56.2) | 1.772 (55.4) | 3.756 (49.6) |
| `  CalculateMAll [40]` | 0.354 (43.0) | 1.444 (45.2) | 3.230 (42.6) |
| `  LocEnergyCal [41]` (`CalHamiltonian1 [71]`) | 0.042 (5.1) | 0.128 (4.0) | 0.195 (2.6) |
| `  ReturnSlaterElmDiff [42]` | 0.044 (5.3) | 0.148 (4.6) | 0.250 (3.3) |
| `  calculate OO and HO [43]` + `multiply store OO [45]` | 0.010 (1.2) | 0.023 (0.7) | 0.028 (0.4) |
| `StochasticOpt [5]` (`calculate S and g [56]`, `DPOSV [57]`) | 0.002 (0.3) | 0.008 (0.3) | 0.017 (0.2) |
| `UpdateSlaterElm [20]` | 0.002 (0.2) | 0.007 (0.2) | 0.014 (0.2) |
| `outputData [22]` | 0.011 (1.3) | 0.018 (0.6) | 0.015 (0.2) |

Per-call-site facts from `MVMC_RS_INNER_PROFILE=1` (serial, one thread):

| call site | L16 | L32 | L64 |
| --- | ---: | ---: | ---: |
| `pfaffian.rs:219` `calc_m_all_real`, us per call (8 planes) | 59 | 241 | 1080 |
| same, us per plane | 7.4 | 30 | 135 |
| `updates.rs:604` `update_m_all_real_flat`, us per accepted move (8 planes) | 2.8 | 8.1 | 29 |
| `updates.rs:116` `calculate_new_pf_m2_real_flat`, us per proposal (8 planes) | 0.23 | 0.39 | 0.76 |
| sampler cost per attempted hop (`hopping update [32]` / proposals) | 2.0 us | 4.3 us | 13 us |

About 36 % of hop proposals are accepted at L32 (69 622 `update_m_all` calls for 192 928
proposals).

### 3.3 Operation table

Classes: **E** einsum/GEMM-expressible; **BF** batched factorization (needs a batched
factorization kernel); **S** inherently sequential; **G** gather/elementwise/indexing;
**R** reduction. Suitability: **High** (large batched dense work, no per-step host dependency),
**Medium** (batchable but gather/launch-bound or small), **Low** (tiny or sequential),
**Host** (stays on the host by design).

Shape and cost columns use the notation of section 2. Costs are flop-order estimates per call,
not measured device times.

| # | operation | file:line | shape and cost | share (L16/L32/L64) | class | GPU suitability |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `calc_m_all_real` (assemble `M`, skew LTL^T `dsktf2`, `utu2pfa`, `utu2inv`, sign flip) per QP | `crates/mvmc-core/src/pfaffian.rs:134`, per-plane `:789`, assemble `:973`; kernels `crates/pfapack/src/ltl.rs:34`, `utu2.rs:52`, `utu2.rs:328` | gather `S[qp][rs_i,rs_j]`: `[2Nsite,2Nsite,NQP]` to `M[n,n,NQP]`; factor `~n^3/3`, inverse `~n^3`; one call per saved sample, batchable over `NQP*NS` independent planes (2 400 planes of `n`=16-64) | 43.0 / 45.2 / 42.6 (`[40]`) + 15.5 / 15.9 / 15.6 (`[34]`, same kernel, sampler recalculation) | BF + G | **High**, but blocked by the missing batched skew factorization/Pfaffian (4.4) |
| 2 | complex and FSZ variants of 1 | `pfaffian.rs:280` (`calc_m_all_complex`), `:417`, `:508` (FSZ complex), `:630` (FSZ real); `crates/pfapack/src/ltl.rs:39` (`zsktf2`), `utu2.rs:57`, `:339` | same, C64, `n` up to `2*Nsite` for FSZ | not in benchmark inputs | BF + G | High (same kernel, C64) |
| 3 | `calculate_new_pf_m2` (rank-one ratio for one proposed hop) | `crates/mvmc-core/src/sampling/updates.rs:26` (complex), `:73` (real), `:168` (fused with IP) | per QP `r_qp = -pf_qp * sum_j invM[qp][msa,j] * S[qp][rsa, rs_j]`: row-dot of length `n`, `NQP` independent; `O(NQP*n)` per proposal | 4.1 / 3.2 / 2.3 | E (batched dot) + G | Medium: trivial flops, launch/sync-bound per proposal; useful only batched over walkers |
| 4 | `update_m_all` (rank-two inverse update after an accepted hop) | `updates.rs:548` (complex), `:588` (real), kernel `update_one_real` `:1064`; two-electron `:703`, `:760` | per QP: `vec1 = -invM * slt` (GEMV), rank-two update of `n x n`; `O(NQP*n^2)` per accepted move | 12.8 / 17.8 / 27.5 (grows as `n^2`) | E (batched GEMV + rank-2 `dot_general` with accumulate) | High **only** with walker batching; sequential across steps within one chain (S) |
| 5 | Metropolis acceptance, candidate generation, `UpdateProjCnt`/`log_proj` | `crates/mvmc-core/src/sampling/metropolis.rs:41`, `:59`; `sampling/driver.rs:122`; `sampling/projection.rs:179`, `:278`, `:393` | scalar per step; RNG draw per valid candidate | 1.9 / 1.6 / 1.4 (`[60]`) + `[31]` 1.4 / 0.7 / 0.3 | S | **Host** (SFMT per chain, C draw order) |
| 6 | `calculate_ip` (QP-weighted sum of Pfaffians) | `crates/mvmc-core/src/observables.rs:128` (real), `:148` (complex) | `sum_qp w_qp pf_qp`, `NQP` terms per sample | in `[40]`/`[3]` | R | High as the tail of 1 (stays on device), single complex scalar down |
| 7 | local energy, transfer (hopping) terms: `green_func1` ratios per term | `observables.rs:2096`, `:3069` (`calculate_local_energy[_timed]`), `green_func1` `:1476`, `refresh_transfer_cache` `:1574`, fast path `:1837` | per term `sum_qp w_qp * new_pf_qp` (item 3 per term); all single-electron-move ratios form a table `A[qp, m, site] = (invM[qp] * S_gathered[qp]^T)`, a batched GEMM `[n,n] x [n,2Nsite]` per QP and sample | 5.1 / 4.0 / 2.6 (`[41]`) | E (table by batched GEMM) + G | Medium-High: more flops than the sparse term loop, but GEMM-friendly; projection factors (`proj_ratio`) per term stay a small gather |
| 8 | two-body Green functions, Lanczos Green | `observables.rs:638` (`green_func2`), `:2541` (`calculate_lanczos_green`), `:2769` (`calculate_lanczos_h2_transfer`); `observables/fsz_green.rs:20`; `lanczos.rs:44`, `:73` (16 moments, host) | PhysCal only: per sample and listed `(i,j,k,l)` element; products of single-move ratios | PhysCal L32: `CalculateGreenFunc` 7.6 % (memo, section 3.1) | E + G | Medium (PhysCal); `lanczos_energy` is host scalar |
| 9 | `slater_elm_diff` (derivative of `ln Psi` w.r.t. Slater parameters, reduce over QP) | `crates/mvmc-core/src/slater_derivative.rs:176`, real fast path `:403`, complex `:456`, FSZ `:473`; reduction helpers `:75`, `:123` | per sample: `NSlater` outputs, each a weighted sum over `NQP` and `n^2`-sparse `invM` entries; structurally `O[orb] = sum_qp w_qp ...` (tenferro einsum `"oq,q->o"` already prototyped, `observables.rs:549`, `dead_code`) | 5.3 / 4.6 / 3.3 | E + G (sparse scatter into `NSlater`) | Medium: scatter/atomic-free reduction needs a deterministic formulation |
| 10 | O vector assembly and store (`set_projection_diff`, `calculate_oo_store[_real]`) | `observables.rs:202`, `:322`, `:426` | writes one column of the `[1+NPara, NS]` store, `ho += w*e*O` | 1.2 / 0.7 / 0.4 (with 11) | G | Low alone; becomes free if 9 writes the store on device |
| 11 | **SR O^T O (S matrix pre-image)**, real | `observables.rs:347` (`finalize_oo_store_real`, `dsyrk`, `[1+NPara, NS]` to `[1+NPara, 1+NPara]`) | `NPara^2 * NS` flops (symmetric rank-k) | included in 10 | **E (SYRK/GEMM)** | High asymptotically (`NPara >~ 3e3`); <0.5 % in benchmarks |
| 12 | SR O^T O, complex (deliberately sequential sum, not einsum) | `observables.rs:447` (`finalize_oo_store`), `:508` (`sr_store_gram_julia`, comment on rounding and signed zeros), `:487` (tensor wrapper) | `4 NPara^2 NS` complex flops | not in benchmark | E | High; blocked on the documented-tolerance decision (6.2) because the CPU path pins the sample-sum order |
| 13 | S matrix and force assembly, diagonal shift | `crates/mvmc-core/src/sr.rs:357` (`collect_active_real`), `:394` (`build_s_g_real`), `:427` (`build_s_g_complex`) | `S_ij = OO_ij - O_i O_j`, `g = -2 dt (HO - E O)`; `n_smat^2` elementwise | 0.1 / 0.1 / 0.1 (`[56]`) | E + G (elementwise on GEMM output) | High asymptotically, trivial now |
| 14 | SR direct solve `dpotrf`+`dpotrs` | `sr.rs:856` (`cholesky_solve`) | `n_smat^3/3` | 0.2 / 0.2 / 0.2 (`[57]`) | BF | High at large `NPara`; tenferro `cholesky`/`triangular_solve` exist, cuSOLVER-backed |
| 15 | SR-CG matvec `z = (O O^H/W) x - mean (mean^T x) + shift*diag*x` | `crates/mvmc-core/src/sr_cg.rs:92` (`stochastic_opt_cg`), `:428` (`apply`, four `dgemv`s), `:264` (`sequential_dot`) | two GEMV per iteration, `2 * NPara * NS` flops each | `NSRCG=1` only (not benchmarked) | E (GEMV) | High at large `NPara*NS`; dots are sequential by contract (C/Julia order) |
| 16 | accumulation across samples/ranks | `crates/mvmc-core/src/average.rs:18`, `:31`, `:50`; `sr_accumulator.rs:143`, `:161`; `run.rs:1750` (`reduce_accumulators`) | elementwise sums over `[1+NPara]`, `NPara^2` | 0.1 | R | Host/MPI; unchanged |
| 17 | Slater table refresh per SR step | `crates/mvmc-core/src/slater_update.rs:17` (`update_slater_elm`), `:128` (FSZ) | `[(2Nsite)^2, NQP]` filled from `NOrbital` parameters by index/sign tables | 0.2 | G | Medium; shares the index tables of item 1, so the device copy of `S` must be refreshed once per step |
| 18 | local-energy diagonal terms | `observables.rs:563` (`calculate_hamiltonian_diagonal`) | `O(Nsite)` per sample | <0.1 (`[70]`) | R | Low |
| 19 | file output, `zvo_*` | `crates/mvmc-core/src/output_files.rs` | host I/O | 1.3 / 0.6 / 0.2 | - | Host |

Existing tenferro use (for reference): storage only for `SlaterElmFlat` and `InvMColMajor`
(`crates/mvmc-core/src/state.rs:43`, `:171`, shapes `[(2Nsite)^2, NQP]` and `[n^2+1, NQP]`
with a trailing pad slot, both asserting host-backed storage), `TypedTensor` buffers in
`slater_derivative.rs:5`, and the einsum Gram/weighted-sum prototypes in `observables.rs`
(`:487`, `:549`) kept off the hot path to preserve C summation order.

### 3.4 Where the speedup is, and where it is not

Upper bounds if a stage costs nothing (Amdahl, shares from 3.2; before transfers/launches):

| offloaded stages (cumulative) | L16 | L32 | L64 |
| --- | ---: | ---: | ---: |
| SR S/g/solve, O store, Slater refresh (items 10-14, 17) | 1.02x | 1.01x | 1.01x |
| + measurement Pfaffian/inverse (item 1, `[40]`) | 1.8x | 1.9x | 1.8x |
| + sampler recalculation (`[34]`) | 2.5x | 2.7x | 2.4x |
| + sampler ratios and rank-two updates (items 3, 4; needs walker batching) | 4.4x | 6.0x | 8.9x |
| + local energy and Slater derivative (items 7, 9) | 8.0x | 12x | 19x |

SR cost model for larger problems (`NPara` = 3 000-10 000, `NS` = 1 000-10 000, not measured
here): the Gram product is `NPara^2*NS` = 1e10-1e12 flops and the Cholesky `NPara^3/3` =
1e10-3e11 flops per SR step, versus about 1e8 at `NPara`=388, `NS`=300. SR is therefore the
natural first target for large-parameter models (for example general-orbital/RBM runs) but not
for the Hubbard-chain benchmark, where it is below 1 %.

Break-even for sampling: the CPU needs 2-13 us per attempted hop on one core. A device round
trip per MC step (kernel launches for ratio, update and weight, plus a device-to-host
download of the acceptance weight so that the host SFMT can decide) is of the order of
tens of microseconds, and a 36-core host already runs 36 independent chains in parallel.
Matching one 36-core socket at L64 (13 us/attempt, about 2.8 M attempts/s) with a 60 us step
needs roughly 170 walkers per device; at L16 (2 us/attempt) over a thousand. These are
estimates from the measured CPU costs and an assumed launch/sync floor, not measured device
numbers; the Phase 3 prototype must measure the floor. The practical conclusion is that the
sampler benefits only with many walkers or with large `n` (where `O(n^2)` rank-two updates
dominate: at L64 `UpdateMAll` is already 27.5 % and growing).

## 4. tenferro-rs survey

Evidence: the registry sources of tenferro 0.7.1 (the version pinned in `Cargo.lock`:
`tenferro-ad`, `-core-ops`, `-cpu`, `-cpu-basic`, `-cpu-fused`, `-einsum`, `-internal-*`,
`-runtime`, `-tensor`, `-tensor-core`), the upstream repository at tag `v0.7.1` and `main`
(2026-10-05) for `tenferro-linalg` and `tenferro-gpu`, and the upstream guides
`docs/guides/devices-and-gpu.md`, `linear-algebra.md`, `choosing-a-backend.md`, `einsum.md`,
`custom-cuda-kernels.md`. No GPU hardware was available, so nothing GPU-side was executed.
Upstream is a pre-1.0 research platform whose README states APIs may change across 0.x.

### 4.1 Crates and local state

* This workspace depends on `tenferro-tensor`, `tenferro-cpu` (`cpu-faer`) and
  `tenferro-einsum` (root `Cargo.toml:53-55`) and carries local lru-bump patches of
  `tenferro-ad`, `-cpu`, `-einsum`, `-runtime` in `third_party/` (issue #192;
  `third_party/README.md`).
* Not in this workspace and not in `Cargo.lock`: `tenferro-gpu` (CUDA/WebGPU/ROCm substrate),
  `tenferro-linalg` (`cholesky`, `solve`, `lu`, `inv`, `det`, ...), `tenferro-fft`,
  `tenferro-xla`. `tenferro-einsum` and `tenferro-ad` expose `cuda`, `webgpu`, `rocm`
  features that pull `tenferro-gpu` (checked in their `Cargo.toml`).
* Enabling a GPU feature would add `tenferro-gpu`, which depends on `tenferro-runtime`,
  `-cpu`, `-tensor`, `-core-ops` and on `lru` (optional, enabled by its GPU features;
  checked in its `Cargo.toml` at `v0.7.1`). It would therefore need the same lru bump as the
  four patched crates (a fifth local patch) unless a fixed release exists (upstream issue
  #1958 tracks the bump itself). Verify `cargo deny` and `cargo tree -d` (one
  `tenferro-runtime` only) when the feature is first added.
* MSRV/edition: tenferro 0.7.1 needs Rust 1.96, edition 2024; the toolchain here is 1.99.

### 4.2 Backend abstraction

* `BackendSession` (trait object, `&mut dyn BackendSession`) and `BackendSessionHost`
  (`with_backend_session(|session| ...)`) are the execution scope. Concrete backends:
  `tenferro_cpu::CpuBackend` (faer default, optional `cpu-blas` with OpenBLAS/MKL/Accelerate,
  `CpuBackend::with_threads_and_kind`), `tenferro_gpu::cuda::CudaBackend::new(CudaDeviceId)`.
* Layering: `TypedTensor<T, R>` (fixed dtype), `Tensor` (runtime dtype; main value for GPU
  dispatch), `EagerTensor`/`EagerRuntime` (`with_cuda_backend`, `synchronize()`), `TracedTensor`
  with `GraphCompiler` and `Runtime::run_compiled` (graph compile and replay).
* Backend traits in `tenferro-tensor` (`src/backend.rs`): `TensorDot` with `dot_general`,
  `dot_general_read`, `dot_general_read_into`, `dot_general_with_conj`,
  `dot_general_read_into_accum`, `dot_general_cached`; `DotGeneralConfig` has
  `lhs/rhs_contracting_dims` and `lhs/rhs_batch_dims`. `TensorStructural::copy_read_into`;
  session ops such as `add_read`, `axpby_read_into_accum`, `fill_zero_write` on CUDA.
* No implicit transfers (PyTorch convention): a CUDA tensor on the CPU backend, or the other
  way round, returns an error. Unsupported op/dtype on CUDA is an error, never a silent CPU
  fallback.

### 4.3 GPU backends

| provider | status (upstream guide) | feature | coverage relevant to mvmc |
| --- | --- | --- | --- |
| CPU | supported | `cpu-faer` (default), `cpu-blas` | all ops |
| CUDA | supported | `cuda` | CubeCL-CUDA kernels plus cuBLAS/cuSOLVER/cuTENSOR; `dot_general` native for F32/F64/C32/C64 with batch, accumulate, conjugation; elementwise add/sub/mul/div/neg/conj for F64/C64; `reduce_sum` F64/C64; structural ops, `gather`/`scatter`/`slice`/`concatenate` for F64/C64; `cholesky`, `triangular_solve`, `lu`, `svd`, `qr`, `eigh`, `solve` for F32/F64/C32/C64 |
| WebGPU | experimental | `webgpu` | F32/C32 `dot_general` only; no F64, no linalg, no elementwise |
| ROCm | not supported for execution | `rocm` reserved | none |

CUDA requirements: CUDA 12.6.2 minimum (12.8 for the full CubeCL feature set), cuTENSOR 2
(2.1.x on compute capability 7.0), cuBLAS/cuSOLVER 12, loaded dynamically; missing libraries
are typed errors. The CUDA table in the guide lists complex `exp/log/sin/cos` as unsupported
and `inv`, `det`, `slogdet` do not appear in the CUDA coverage tables (they are part of the
concrete linalg surface; their CUDA status must be verified before relying on them).
The `full_piv_lu`, general `eig` and `dynamic_update_slice` ops have no CUDA implementation.

Host-side tests for the GPU paths in the upstream repo are `#[ignore = "requires CUDA"]`
integration tests (for example `tests/integration/cuda_eager_tensor.rs`,
`gpu_ad_tests.rs`), run on a gated GPU lane.

### 4.4 Linear algebra: Pfaffian, skew factorization, inverse, batching

* Concrete surface (`TensorLinalgExt`, `TypedTensorLinalgExt`, `TensorReadLinalgExt`):
  `solve`, `triangular_solve`, `cholesky`, `svd`/`svdvals`, `qr`, `householder_qr`
  (incremental), `eigh`, `eig`, `lu`, `full_piv_lu[_solve]`, `pinv`, `det`, `slogdet`, `inv`,
  `norm`. All take `&mut dyn BackendSession`.
* **There is no skew-symmetric factorization, Pfaffian, or LTL^T/Parlett-Reid op.** A search of
  the upstream tree and issues for "pfaffian" and "skew-symmetric" finds only unrelated
  results. `slogdet` returns sign and `log|det|`; since `Pf(A)^2 = det(A)`, only `|Pf|` is
  recoverable and the Pfaffian sign is lost, so it cannot replace `dsktf2`+`utu2pfa`: mVMC
  sums signed Pfaffians over QP planes.
* Batching convention: **trailing** batch dims in column-major storage
  (`[M, N, B...]`; design note `docs/superpowers/specs/2026-04-05-batched-linalg-design.md`),
  each batch slice contiguous; shapes `cholesky [N,N,B..]`, `solve A [N,N,B..], b [N,M,B..]`,
  `svd`, `qr`, `eigh` as listed there. The batch convention for `lu`, `inv`, `det`,
  `slogdet` and the CUDA batched paths (the guide mentions `cusolverDnXsyevBatched` for `eigh`)
  should be confirmed per op with a test before use (the design note lists only
  cholesky/svd/qr/eigh/solve).
* Inverse: `inv` is a general (LU/solve-based) inverse. It ignores skew structure
  (about 2x flops and no structural exactness) and its rounding differs from `utu2inv`.
  Usable as a cross-check, not as the production inverse.
* GEMM/GEMV shaped linear algebra needed by SR and CG is `dot_general` (batched, accumulate,
  conjugation) and the einsum front end: sufficient.

### 4.5 Einsum

* Typed and concrete einsum: `TypedTensorEinsumExt`/`TensorEinsumExt` (`[&a,&b].einsum("ij,jk->ik", session)`),
  `einsum_into`, `einsum_read[_into]` (borrowed `TensorRead` views), `einsum_notation`,
  and `ConcreteEinsumPlan::prepare_read` / `execute_read_into` for repeated fixed
  subscripts/dtypes/shapes. Explicit arrow required; `...` ellipsis supported in flat
  notation; N-ary contraction order chosen automatically (omeco) with an explicit-path option.
* Complex einsum is supported (`TypedTensor<Complex64>`); batch labels such as
  `"qij,qjk->qik"` lower to `dot_general` batch dims (the upstream guide documents the binary-einsum lowering
  explicitly for WebGPU; CUDA `dot_general` is native).
* The repo's own Gram prototype comment (`observables.rs:504`) records the relevant issue: a
  general einsum backend is free to change rounding and signed zeros, so the CPU C-order path
  must keep its explicit loops; einsum is only for accelerated paths.
* No recognition of the symmetric `A A^T` (SYRK) pattern is documented; the product costs
  2x the flops of `dsyrk`.

### 4.6 Complex dtype, determinism, transfer

* Complex: `C32`/`C64` are first-class dtypes for tensors, einsum, `dot_general` and linalg
  (CPU and CUDA). Complex `exp/log/trig` are unsupported on CUDA (not needed on the device
  by mvmc: log/exp of the weights stay on the host).
* **Determinism and reduction order: not documented.** A search of the upstream guides and
  the 0.7.1 sources finds no contract for the summation order of `dot_general`/`reduce_sum`,
  no deterministic-mode flag, and no statement about run-to-run reproducibility on CUDA
  (library algorithm and workspace choices may differ; not verified). `choosing-a-backend.md` says
  the faer provider is "the default choice when placement, reproducibility, and predictable
  nesting matter", which only covers CPU thread placement. External BLAS providers own their
  thread pools; CPU results can differ between faer and OpenBLAS.
* Host/device transfer: explicit `tenferro_gpu::cuda::upload_tensor(backend.runtime(), &cpu)`
  and `download_tensor(backend.runtime(), &gpu)`; `cuda_devices()`, `gpu_available()`.
  Eager CUDA launches are asynchronous; the host waits at download or host inspection;
  `EagerRuntime::synchronize()` is the explicit barrier. Host slice access on a GPU tensor
  (`host_data()`, `as_slice`) is an error. No pinned-memory API, no asynchronous
  download with an event handle, and no partial/sub-region download are documented.
  `SlaterElmFlat`/`InvMColMajor` call `host_data().expect(...)`, so today they are host-only
  by construction.
* Extension point for missing kernels: `tenferro_gpu::cuda::raw` /
  `CudaBackend::with_backend_session -> with_cuda_exec_session -> CudaExecSession::with_raw`
  (load PTX/CUBIN or NVRTC, launch on the session stream, tensor-owned allocations); upstream
  issue #1597 designs a public external kernel API across CUDA, CubeCL, WebGPU, ROCm and
  multi-GPU. This is the supported way for mvmc to add a Pfaffian kernel without forking
  tenferro.

### 4.7 Gaps and draft upstream feature requests (not filed)

Ordered by value to mvmc-rs.

1. **Batched skew-symmetric LTL^T (Parlett-Reid) factorization, Pfaffian and skew inverse**
   (`skew_ltl`, `pfaffian`, `skew_inv`; F64 and C64; trailing batch dims `[N,N,B..]`;
   CPU and CUDA; returns Pfaffian value (or sign plus log-abs) and a zero-pivot status per
   batch element rather than failing the batch). Reference semantics: PFAPACK
   `dsktrf/zsktrf`, `dsktri` (`extern/mVMC-1.3.0/src/pfapack`), `n` = 16-1 024, batches of
   1e3-1e5.
2. **A reduction-order and determinism contract.** Document the summation order of CPU
   `dot_general`/`reduce_sum` (per provider), and add an execution policy (for example
   `ExecutionPolicy::Deterministic`) that on CUDA forbids split-K/atomics and fixes
   algorithm and workspace selection, guaranteeing bitwise run-to-run reproducibility for a
   fixed device, driver, library versions and shapes. Expose the selected algorithm so that
   tests can record it.
3. **CUDA coverage and batching documentation and tests for `inv`, `det`, `slogdet`, `lu`
   (batched), `solve` with many right-hand sides.** State the batch axis convention per op in
   the guide and add a coverage table row for each.
4. **Symmetric rank-k / Gram contraction.** Recognize `"ij,kj->ik"` with the same operand (or
   expose `syrk`) on CPU and CUDA (cuBLAS `dsyrk/zherk`), halving the SR S-matrix flops, and a
   `cholesky_solve` (`potrs`) helper over a stored factor.
5. **Asynchronous host transfer.** Pinned host buffers; `download_tensor_async` returning an
   event/future; sub-region download (for example one scalar per batch element) so a
   walker-batched sampler downloads `B` weights per step instead of whole tensors; and
   upload of a small index tensor into a preallocated device buffer.
6. **Replay of small fixed op sequences with low launch overhead.** CUDA-graph capture or an
   equivalent prepared-sequence API for a fixed-shape session (gather, batched GEMV, rank-2
   update, weight reduction) so that per-step launch cost is a single graph launch. The
   existing `ConcreteEinsumPlan` covers one einsum only.
7. **Gather/index ergonomics for batched two-index gather**: `A[b, idx_i[b,k], idx_j[b,l]]`
   in one op, and `I64` operands for `gather`/`scatter` on CUDA (currently F32/F64/I32/Bool/
   C32/C64 operands only; `I64` operands not implemented).
8. **Complex `exp`/`log` on CUDA** (low priority; mvmc keeps these on the host).
9. **Publish tenferro with `lru >= 0.18.5`** (open upstream issue #1958) so that the local
   `third_party/` patches (issue #192) can be removed before adding `tenferro-gpu`.

## 5. Design

### 5.1 Principles

* **Authority.** The C-order CPU implementation is the numerical and RNG reference. Any
  accelerated implementation is checked against it (section 5.6); nothing in the CPU path is
  replaced, reordered or made generic over a tensor backend.
* **Stage-level dispatch.** The backend trait operates on whole stages over batches, never
  on scalar loops, so that launch and transfer costs amortize and the CPU implementation
  remains the existing hand-written loops.
* **Device residency.** State that is read and updated every MC step (`S`, `invM`, `pf`)
  lives on the device between steps. The host sends proposals and receives scalar
  weights; it never copies planes.
* **No GPU in normal CI.** GPU code is behind cargo features and optional gates; CPU tests
  neither compile nor link tenferro GPU crates. Per `AGENTS.md`, nothing here changes
  `c_toolbox/` independence.
* **No silent fallback.** Following tenferro, an unsupported op on a selected device is an
  error; the runner chooses CPU or accelerated explicitly (`MVMC_BACKEND`/CLI option,
  default CPU).

### 5.2 Backend abstraction in mvmc-core

A single trait in a new module `crates/mvmc-core/src/backend/` (name provisional), implemented
by `CpuCOrder` (the current code, unchanged) and by `Tenferro<B>` (accelerated, feature
gated). Method granularity (all take batch dims explicitly):

```text
trait VmcBackend {
    // state that lives with the backend (host Vec for CpuCOrder, device tensors otherwise)
    type SlaterTable;   // S      [(2 Nsite)^2, NQP]
    type InverseSet;    // invM   [n, n, NQP, B], pf [NQP, B]
    type SampleStore;   // O      [1 + NPara, NS]

    fn upload_slater(&mut self, host: &SlaterElmFlat<f64>) -> Self::SlaterTable;        // once per SR step
    // (1) Pfaffian + inverse for B configurations x NQP planes; status per plane
    fn pfaffian_inverse(&mut self, s: &Self::SlaterTable, ele_idx: &[i64] /* [n,B] */)
        -> (Self::InverseSet, BatchStatus);
    // (3)+(5) proposals: B moves -> B scalar weights, one device->host transfer
    fn propose_hops(&mut self, inv: &Self::InverseSet, s: &Self::SlaterTable,
                    moves: &[Hop] /* B */, qp_weights: &[Complex64]) -> Vec<Complex64>;
    // (4) apply accepted moves only (mask of B)
    fn apply_hops(&mut self, inv: &mut Self::InverseSet, s: &Self::SlaterTable,
                  moves: &[Hop], accepted: &[bool]);
    // (7)(8)(9) measurement tables for NS saved configurations
    fn local_energy_table(&mut self, ...) -> Vec<Complex64>;
    fn slater_diff(&mut self, ...) -> Self::SampleStore;
    // (11)(13)(14)(15) SR
    fn sr_gram(&mut self, o: &Self::SampleStore, w: &[f64]) -> SrMatrices;
    fn sr_solve(&mut self, s: SrMatrices, g: &[f64]) -> Result<Vec<f64>, SrError>;
    fn sr_cg_matvec(&mut self, ...) -> ...;
}
```

Where tenferro tensors flow:

* `SlaterTable`, `InverseSet`, `SampleStore` are `tenferro_tensor::Tensor` (runtime dtype,
  F64 or C64) owned by the accelerated backend; `TypedTensor` stays the host-side storage
  for the CPU path (`state.rs:43`, `:171`), which already exposes column-major buffers.
* SR matvec/Gram/force and the local-energy tables use `dot_general`/einsum
  (`DotGeneralConfig` batch dims over QP and sample).
* The Pfaffian/inverse uses a kernel that tenferro does not have yet (4.4): either the
  upstream op (request 4.7-1) or an external CUDA kernel launched through
  `CudaExecSession::with_raw` (4.6) behind the same trait method.
* Host stays authoritative for: SFMT draws, candidate generation, Metropolis decision, proj
  counts, parameter update order, file output, MPI reduction.

Dispatch granularity is chosen per stage, not per operation: the runner calls
`backend.pfaffian_inverse` once for all `NS*B` saved configurations of an SR step (one launch
of `NQP*NS*B` planes) instead of once per sample.

### 5.3 CPU C-order path stays authoritative

* `CpuCOrder` keeps the current functions (`calc_m_all_real`, `update_m_all_real_flat`,
  `calculate_new_pf_m2_real_flat`, `dsyrk`, `dpotrf`, `dgemv`) verbatim with their
  operation order (`#358`) and Rayon inner threading (`#360`, `#361`).
* The existing worker-invariance tests, fixtures and 20-step repeatability tests continue to
  run only against `CpuCOrder`, and the default build/tests never select the accelerated
  backend.
* The accelerated backend may reorder sums (GEMM tiling, parallel reductions, different
  Pfaffian algorithm structure) under the tolerance policy of section 5.6. It must never
  change RNG draw order or count (draws live on the host), and may change Metropolis
  decisions only through a documented, located numerical acceptance divergence.
* tenferro's CPU backend (faer) is **not** a substitute for the reference. It is used in the
  accelerated path's CPU smoke tests (the same code on `CpuBackend`), which lets the
  accelerated code be verified without a GPU.

### 5.4 Data layout

* Column-major throughout (tenferro and the existing `InvMColMajor`/`SlaterElmFlat`), batch
  dimensions **trailing** (tenferro convention), so a batch slice is contiguous.
* Device layout (accelerated path only):

| tensor | shape (col-major) | note |
| --- | --- | --- |
| `S` Slater table | `[2Nsite, 2Nsite, NQP]` | same bytes as `SlaterElmFlat` `[(2Nsite)^2, NQP]`; refreshed once per SR step (item 17) |
| electron sites | `[n, B]` (i64 or i32) | `ele_idx` per walker; host-built, small |
| `M`/`invM` | `[n, n, NQP, B]` | **no pad slot**: the CPU layout `[n^2+1, NQP]` appends a Pfaffian pad per QP (`state.rs:179`); the batched layout keeps `pf` separately as `[NQP, B]` so the planes reshape to tenferro batch dims without a copy. A host conversion helper maps between the layouts for tests |
| `pf` | `[NQP, B]` | |
| QP weights | `[NQP]` (C64) | `w_qp` for `calculate_ip` |
| O store | `[1 + NPara, NS]` (F64 or C64) | already the existing `sr_opt_o_store*` layout `[component, sample]` |
| SR matrices | `[n_smat, n_smat]`, `[n_smat]` | `n_smat <= NPara` after the cut |

* Assembly of `M[qp]` from `S` is two index gathers with host-built index vectors (the
  `rs_i = ele_idx[i] + spin*Nsite` map of `pfaffian.rs:973`), executed on the device.
* Precision: F64/C64 only. F32/C32 (the only WebGPU precision) is not acceptable for the
  Metropolis weights and Pfaffians at the target tolerances; a mixed-precision sampler is out
  of scope until the F64 path is characterized.
* Memory per SR step (F64): `invM` for all saved samples `n^2 * 8 B * NQP * NS` = 4.9 MB /
  19.7 MB / 79 MB for L16 / L32 / L64 at `NQP=8`, `NS=300`, times `B`; O store
  `(1+NPara)*NS*8 B` under 1 MB per chain. A walker batch of `B` = 256 at L64 would need
  `B * 79 MB` (about 20 GB) only if all samples of all chains were held at once, so
  measurement is processed in chunks of samples.

### 5.5 Walker and sample batching, seeds, step protocol

Two independent batch axes:

* **Samples inside a chain** (`NS`): the saved configurations of one chain are independent
  for measurement. The loop in `crates/mvmc-core/src/run.rs:2828` (`accumulate_observables_local`)
  currently runs `calc_m_all` -> IP -> local energy -> Slater derivative -> O store once per
  sample; it can be restructured into batched stages over `NS` configurations
  (`NQP*NS` planes). This needs no change to RNG or to the sampling trajectory, only to the
  order of the measurement stages (and is therefore a pure reordering of independent work on
  the CPU path as well).
  **Implemented in #422** (`crates/mvmc-core/src/measurement_batch.rs`, loop in
  `run.rs::accumulate_observables_local`): stage A builds the Pfaffian/inverse tables for `B`
  samples, stage B runs the unchanged per-sample consumers in sample order, so all
  accumulation orders and outputs are independent of `B`
  (`MVMC_RS_MEASURE_BATCH`, default 4; `B = 1` is the former serial order). All modes are
  covered (real, complex, FSZ, DH, RBM, OptTrans, PhysCal Green and Lanczos). The CPU slots
  keep the working layout and are swapped into the state (no copy); `BatchedPlanes::pack`
  produces the device layout `[n,n,NQP,B]` / `[NQP,B]` (trailing batch, column-major). A first
  variant that copied every sample into and out of that layout cost 2.7 % on L32, so it is
  not on the CPU hot path. Local energy, Green functions and the Slater derivative are still
  per-sample in stage B (phase 4).
* **Chains (walkers)** (`B`): Metropolis sampling is sequential within a chain. Independent
  chains, each with its own SFMT stream, are the only way to fill a device during sampling.

C seed contract (authoritative): `init_gen_rand(RndSeed + group1)` with
`group1 = rank0 / NSplitSize` (`extern/mVMC-1.3.0/src/mVMC/vmcmain.c:239-257`; Rust:
`resolve_rnd_seed`, `run.rs:1976`, `base.wrapping_add(group1)`, then `seeded_rng`). Ranks
inside one group share one stream and split the QP range; groups are independent chains.
The accelerated sampler therefore defines **walker `w` of a device as global group
`group_base + w`, seed `RndSeed + group_base + w`**, and every walker owns a host
`Sfmt19937Rng`. A run with `B` walkers on one device is, in terms of RNG, identical to a run
with `B` MPI groups of `NSplitSize` ranks, and each walker is reproducible against the CPU
implementation seeded the same way. `NSplitSize > 1` (QP split inside a group) maps to the
QP batch dimension of the same walker.

Step protocol for `B` walkers (the host loop stays in C order per walker):

1. For each walker the host draws update type and candidate exactly as `vmc_make_sample_*` does
   (`sampling/driver.rs:122`) and builds `Hop`s; walkers whose candidate is rejected before the
   weight evaluation draw nothing more, as in C.
2. `propose_hops` evaluates all valid proposals as one batched stage (`NQP` ratios each, QP
   weighted sum on device) and returns one complex scalar per walker.
3. For each walker the host computes the acceptance weight, draws `genrand_real2()` (the draw
   is unconditional for every valid candidate, `metropolis.rs:59`) and decides.
4. `apply_hops` applies the accepted subset (mask) as one batched rank-two update.

One device synchronization per step for all `B` walkers. Draw order and count per walker are
those of the CPU path given the same decisions; walkers do not share RNG state, so batching
cannot change any walker's stream. Steps stay in lockstep only for the device; walkers may
have different numbers of valid proposals in a step (masks), which affects throughput only.

### 5.6 Validation and tolerance strategy

All accelerated-vs-CPU comparisons use the explicit absolute/relative bounds approach of
`docs/NUMERICAL_COMPARISONS.md` (`|a-e| <= abs + rel*max(|a|,|e|)`, justified per operation,
no bitwise comparison of computed floating-point values).

Levels:

1. **Kernel tolerance tests** (fixed inputs, run on tenferro CPU backend in normal CI and on
   CUDA in the optional gate). Pfaffian: relative bound scaled by `n` and the Pfaffian
   conditioning, plus the identity `Pf(A)^2 = det(A)` as an independent check; inverse: the
   residual `||A * Ainv - I||` (independent of the CPU algorithm) and the skew structure
   `Ainv^T = -Ainv`; GEMM/einsum: operation- and dimension-dependent epsilon budgets as in the
   existing SR rows of the NUMERICAL_COMPARISONS table (sampled O/HO and Gram `1e-12`,
   solver budgets by dimension). Compare also to the CPU C-order result, not only to a
   tenferro CPU result.
2. **Teacher-forced trajectory replay.** The accelerated backend consumes the CPU
   trajectory's decisions (the accepted/rejected sequence and configurations) and reports
   the maximum deviation of `pf`, `invM`, ratios, weights, O store, S, g at every step or
   every K steps, so numerical drift is measured without a branching trajectory.
3. **First-divergence location for free-running comparison.** For each valid proposal record
   the acceptance weight, the draw and the decision margin `|w - draw|`. A decision flip is
   admissible only if the margin is below the demonstrated weight error and is reported with
   its step/walker/operation; any flip with a larger margin is a defect, as is any difference
   in RNG draw count on identical control paths. After a flip the trajectory is allowed to
   diverge (per `AGENTS.md`); comparison then reverts to same-implementation repeatability.
4. **Same-implementation reproducibility.** Accelerated run, same seed, same device and
   library versions: reproducible to the stated tolerance (bitwise only if upstream request 4.7-2
   lands and is verified; it is not assumed). The existing 20-step repeatability style is used.
5. **Statistical checks** (energy within error bars over many walkers) supplement 1-4 and
   never replace them.

RNG contracts (initialization, draw order and count, conversion, state) are tested on the host
sampler independent of the backend and must hold with the accelerated backend selected.
Tolerances are chosen from the first measured divergence and the problem scale, not tuned
upward; a tolerance must never hide an algorithm or RNG defect. References remain Linux
x86_64 for numerical reference; GPU results are labelled with device model, driver, CUDA and
cuBLAS/cuSOLVER versions and the tenferro version.

### 5.7 Build, features and CI

* Cargo feature `gpu-cuda` (mvmc-core, mvmc-cli) marks that a CUDA provider may be
  registered; it adds no dependency. The CUDA dependency tree lives in the standalone
  workspace `gpu/mvmc-gpu-cuda` (decision and measurements in section 10). Off by default; no
  default-feature or lock-file change.
* The accelerated implementation is also compilable with the tenferro **CPU** backend
  (feature-independent module), so most of it is tested in normal CI.
* GPU execution is an optional gate like `.github/workflows/optional-gates.yml` (manual or
  self-hosted GPU runner), with `TENFERRO_REQUIRE_CUDA=1`-style hard failure when the gate is
  requested, and an explicit "skipped, no device" status otherwise (never a silent pass).
* Reference fixtures stay in `tests/fixtures/` and are produced separately; no test reads
  `c_toolbox/` (AGENTS.md).

## 6. Phased roadmap

Benefits are upper bounds from section 3.4 (before launch/transfer costs), for the Hubbard
benchmark class. "Needs" lists blockers.

| phase | scope | deliverable | expected benefit | needs |
| --- | --- | --- | --- | --- |
| 0 | design and inventory (this document) | `docs/design/gpu-readiness.md` | none | none |
| 1 | pathfinder: SR O store Gram, S/g assembly, Cholesky solve and CG matvec through tenferro `dot_general`/`cholesky`/`triangular_solve` on `CpuBackend`, behind the backend trait with a CPU-equivalence test | trait skeleton, device-residency API, first accelerated stage | 1.01x on the benchmark (SR < 1 %); large for `NPara` >~ 3e3 (Gram 1e10-1e12 flop/step) | none (tenferro 0.7.1 suffices) |
| 2 | batched `[n,n,NQP,B]` Pfaffian/inverse: sample-batched measurement stage; accelerated implementation through an external CUDA kernel or upstream op; CPU reference of the batched layout | `pfaffian_inverse` on CPU (reference) and CUDA | 1.8-1.9x (`[40]`), 2.4-2.7x with the sampler recalculation (`[34]`) | request 4.7-1 or an in-repo kernel via `CudaExecSession::with_raw`; request 4.7-5 helps |
| 3 | walker batching: multi-chain runner with `RndSeed + group` streams and the step protocol of 5.5; batched `propose_hops`/`apply_hops` | `B`-walker sampler | 4.4-8.9x (sampler ratios and rank-two updates); only profitable at `B` of the order of 1e2-1e3 | phase 2; request 4.7-5, 4.7-6 to cut per-step latency |
| 4 | local energy tables (batched GEMM `A[qp,m,site]`), Slater derivative and O-store on device; PhysCal Green functions | measurement stages on device | 8-19x cumulative (items 7, 9 plus all above) | phase 2; request 4.7-7 helps |
| 5 | multi-device and MPI interplay (one device per rank, group seeds across ranks), mixed precision investigation | scaling study | scale-out | phases 2-4 |

Gate for every phase: CPU path fixtures unchanged; accelerated stage matches the CPU path
within the section 5.6 tolerances; no change to the default build; optional gate recorded with
device/driver/library versions.

## 7. Proposed follow-up issues (to be filed)

Ordered by expected benefit; phase dependencies in parentheses.

1. Batched Pfaffian + inverse for `NQP*NS*B` planes on a GPU (phase 2): the largest single
   benefit (58-61 % of wall time at the benchmark sizes).
2. Sample-batched measurement pipeline on the CPU path with a batched layout and the backend
   trait (phase 2 enabler; pure reordering of independent work).
3. Multi-chain/walker batching runner with per-chain SFMT seeds `RndSeed + group` and
   one-sync-per-step protocol (phase 3).
4. Local-energy single-move ratio tables by batched GEMM and device Slater derivative (phase 4).
5. SR pathfinder: S/g/Cholesky/CG through tenferro on `CpuBackend` with a CPU-equivalence test
   (phase 1; lowest risk, first to land, small benefit on the benchmark).
6. Accelerated-backend validation harness: teacher-forced replay and decision-margin
   reporting, and the corresponding section in `docs/NUMERICAL_COMPARISONS.md`.
7. `gpu-cuda` feature and optional CUDA gate, including the `third_party/` lru patch
   interplay with `tenferro-gpu`.

(Items 1-2, 5-7 are independent enough to start in parallel; the sequence by dependency is
7, 5, 2, 1, 6, 3, 4. Full problem/acceptance text is in the pull request description.)

## 8. Risks and open questions

* The central risk is the missing Pfaffian kernel: an in-repo CUDA kernel (via
  `CudaExecSession::with_raw`) is a large piece of numerical work (pivoted skew LTL^T with
  panel updates); an upstream tenferro op would remove it from this repository.
* Latency-bound regimes (small `n`, few walkers) are slower on a GPU than on one CPU core;
  the roadmap measures the launch/sync floor in phase 3 before committing to the sampler.
* A deterministic mode is not available upstream; reproducibility on GPU is "same device,
  same library versions, to tolerance" until request 4.7-2 lands.
* Pivoting makes the factorization data-dependent: zero-pivot handling (status per plane, C's
  `info`/abort behavior and the "all zero" and non-finite checks of `pfaffian.rs:789`) must be
  expressed as per-plane status without failing the batch.
* The numbers in sections 3 and 5.5 are single-run, one-thread, shared-host measurements;
  device-side costs are estimates until measured on hardware.
* Mixing the patched `third_party/` tenferro crates with a registry `tenferro-gpu` needs a
  check that Cargo resolves a single `tenferro-runtime` and that the `lru` advisories stay
  closed (issue #192).

## 10. `gpu-cuda` build and optional gate (issue #420)

Status: implemented and run on hardware (2x RTX 3060, Linux x86_64).

### 10.1 Decision: standalone opt-in workspace, not an optional dependency

An optional `tenferro-gpu` dependency in `mvmc-core` was tried first. Cargo records optional
dependencies in `Cargo.lock`, and `tenferro-gpu/cuda` adds about 2000 lines (cubecl, cudarc,
`lru` 0.12 and others), changing the lock file and the audit surface of every default build.
The acceptance criterion forbids that, so:

* `mvmc-core` gains the always-compiled module `backend` (CPU always available, `BackendKind`,
  `DeviceReport`, `device_report`, the gate decision `cuda_gate_decision`) and the feature
  `gpu-cuda = []` (forwarded by `mvmc-cli`) that enables the `CudaProvider` registry. Without
  the feature (or without a registered provider) `BackendKind::Cuda` is an error: no silent CPU
  fallback. `cargo check -p mvmc-cli --features gpu-cuda` needs no CUDA toolkit.
* `gpu/mvmc-gpu-cuda` (own `[workspace]`, own `Cargo.lock`, excluded from the root workspace
  like `benchmark/pfapack_compare`) holds the `tenferro-gpu` dependency, registers the
  provider, queries versions and holds the gate. `Cargo.lock` of the root is unchanged.
  The cost is that `cargo build --features gpu-cuda` at the root does not by itself link CUDA;
  GPU work is built from `gpu/mvmc-gpu-cuda` (downstream crates such as the batched
  Pfaffian can depend on it the same way).

### 10.2 `lru` and the patches

`tenferro-gpu` 0.7.1 requires `lru ^0.12` (optional, enabled by its `cuda` feature), the same
vulnerable range as the four crates patched for #192. The issue is real: without a patch the
GPU lock would contain `lru` 0.12.5. `third_party/tenferro-gpu` is a fifth snapshot with only
the `lru` requirement raised to `0.18.5` (see `third_party/README.md`), and the standalone
workspace repeats all five patches. Verified in `gpu/mvmc-gpu-cuda/Cargo.lock`: one `lru`
(0.18.5) and one `tenferro-runtime` (0.7.1), and it compiles unmodified against lru 0.18.
Remove the snapshot with the other four when upstream tenferro publishes an advisory-safe
release (upstream tenferro #1958).

Version trap from the `tenferro-decision-rs` survey: crates.io 0.7.1 and the source revision
pinned there differ in the `with_backend_session` signature (tenferro #1971). This repository
and the GPU workspace both use crates.io 0.7.1, so `tenferro-gpu` is pinned consistently.

### 10.3 Gate and report

* `scripts/run_cuda_gate.sh [native|docker]` runs the ignored test
  `gpu/mvmc-gpu-cuda/tests/cuda_gate.rs` with `MVMC_RS_CUDA_GATE=1`. `docker` uses an NVIDIA
  CUDA toolkit image with `--gpus all` (default `tenferro-benchmark-cuda:full-verify-20260822`,
  CUDA 12.9.2; override with `MVMC_RS_CUDA_IMAGE`) and the host rustup/cargo.
* `MVMC_RS_CUDA_GATE=1` and no device: hard failure. Unset and no device:
  `cuda-gate: ExplicitSkip: skipped, no device (...)`, not a pass. Both were exercised in a
  container without `--gpus`.
* The report records device, compute capability, memory, driver (NVML), CUDA driver API,
  NVRTC, cuBLAS, cuSOLVER and tenferro versions (`DeviceReport`; unavailable entries print
  `unavailable`). tenferro does not expose these, so versions are read by `dlopen`; no toolkit
  is needed at build time.
* CI: `optional-gates.yml` input `cuda_gate` dispatches job `cuda-gate` on a self-hosted
  runner (`[self-hosted, linux, x64, gpu]`, placeholder labels). It is outside the bounded
  ledger like `mpi-explicit`: NotRun when not selected, fails closed without a device. Hosted
  runners have no GPU, so the default and `plan`/`bounded` paths are unaffected.
* Linking tenferro can exhaust hosted-runner disk (tenferro-decision-rs): the GPU workspace
  sets `debug = 0`.
* Behaviour follows the survey: unsupported op or dtype is a backend error, never a CPU
  fallback; stages are timed separately with `synchronize` inside the timed compute region.

### 10.4 Measured

RTX 3060, driver 580.178.04, CUDA driver API 13.0, toolkit 12.9, cuBLAS 12.9.2, cuSOLVER
11.7.5, tenferro 0.7.1, Linux x86_64, 36 host threads.

Median of 7 runs after one warm-up, milliseconds, through tenferro `EagerRuntime` on both
backends (the CPU column is tenferro's faer CPU backend, not the BLAS/LAPACK path of
`mvmc-core`). Upload/download are pageable-host transfers. FP64 on this consumer GPU runs at
about 1/64 of FP32. Tolerances are relative max-norm vs tenferro CPU: dot 1e-11, Cholesky
(lower triangle, diagonally dominant input) 1e-10; the largest observed errors were 5.2e-15
and 8.9e-16, so the margins are large and set from the scale `n * eps`, not tuned. Single-run
shared-host numbers; CPU times in particular are noisy.

| op | dtype | n | CPU compute | CUDA upload | CUDA compute | CUDA download | CPU/CUDA compute |
|---|---|---|---|---|---|---|---|
| dot | f64 | 256 | 9.5 | 0.83 | 0.37 | 0.25 | 25x |
| dot | c64 | 256 | 7.6 | 1.46 | 0.83 | 0.41 | 9.1x |
| cholesky | f64 | 256 | 65.6 | 0.55 | 1.57 | 0.30 | 42x |
| cholesky | c64 | 256 | 30.5 | 0.72 | 1.48 | 0.34 | 21x |
| dot | f64 | 1024 | 20.3 | 31.7 | 13.7 | 6.8 | 1.5x |
| dot | c64 | 1024 | 45.9 | 59.9 | 38.1 | 13.5 | 1.2x |
| cholesky | f64 | 1024 | 315.5 | 6.2 | 5.5 | 2.1 | 57x |
| cholesky | c64 | 1024 | 168.2 | 37.4 | 16.0 | 7.5 | 10x |
| dot | f64 | 2048 | 93.6 | 156.2 | 96.5 | 25.2 | 0.97x |
| dot | c64 | 2048 | 223.7 | 348.4 | 300.6 | 44.9 | 0.74x |
| cholesky | f64 | 2048 | 347.1 | 77.8 | 26.2 | 24.3 | 13x |
| cholesky | c64 | 2048 | 417.3 | 157.7 | 79.6 | 45.8 | 5.2x |

Reading: the FP64 GEMM at n >= 1024 is at parity with the 36-thread CPU (FP64 throughput of
the card), Cholesky is faster at n >= 256, and for these sizes upload costs as much as
compute, so device residency across stages (section 5) is required for any benefit. At
n = 64 the CPU wins (Cholesky 0.27 ms vs 0.60 ms on the device). The CPU path remains the
reference; this is a feasibility measurement, not a speed-up claim for mvmc.

### 10.5 Validation harness (issue #424)

`mvmc_core::accel_validation` implements the validation policy of section 5.6: teacher-forced
replay against the C-order oracle (pf, invM, weights, O store, S, g), per-proposal
weight/draw/margin recording with a decision-flip defect rule, 20-step repeatability and the
benchmark metadata block. It drives the unified `StageBackend` object of section 14 (a stage
that is not implemented reports `Unsupported`, never a CPU fallback). The CPU tenferro variant
runs in normal CI; the CUDA variant is the second ignored test of the CUDA gate (CUDA RTX 3060
run: S and g match the oracle, 0 flips, 0 defects, Pfaffian stage unsupported). The batched
Pfaffian backends of #423 plug in through `mvmc_gpu::stages::BatchedStages` (section 11.6). The policy text is
in `docs/NUMERICAL_COMPARISONS.md`.

### 10.6 Multi-chain walker runner and device break-even (issue #425)

`mvmc_core::multichain` runs `W` independent PhysCal chains side by side
(`run_phys_cal_multichain`). The contract:

* **Seeds.** Walker `w` uses `RndSeed + group_base + w`, the C group seed
  (`init_gen_rand(RndSeed + group1)`, `resolve_rnd_seed`): `W` walkers on one process are the
  chains of `W` groups of a grouped C run. Each walker owns its host SFMT stream; a time-based
  seed (`RndSeed < 0`) is resolved once for all walkers (the C group broadcast).
* **Draw order and count.** Nothing is added, removed or reordered: a walker is the existing
  serial run with seed offset `group_base + w`. Tests (`crates/mvmc-core/tests/multichain.rs`)
  compare, for every walker, the complete final SFMT state (624 words and position), the number
  of consumed words, the recorded Metropolis `(weight, draw)` sequence and the whole sampling/
  observable state (`Debug` bytes) against that serial run; `W = 1` equals the plain serial
  run, and results do not depend on the worker pool size.
* **Thread budget.** One rayon pool over walkers (`threads`, default `min(W, cores)`), each
  walker single-threaded, BLAS pinned to one thread. A walker owns all of its working state
  (tables, scratch, RNG), the `tenferro-decision-rs` workspace pattern; nothing is shared
  mutably.
* **Batched stages.** `pack_walker_tables` packs the Pfaffian/inverse tables of all walkers in
  the `[n, n, NQP, W]` `BatchedPlanes` layout of #422 (walker `w` in slot `w`, tested against the
  walkers' own tables): the staging format for the batched Pfaffian/inverse of #423. Lock-step
  recomputation of the tables across walkers inside the sampler (one batched call per
  recalculation) is not wired: it only pays with device-resident planes (below), and the
  batched engine of #423 (PR #430) was not merged when this was written.
* **Decision margins.** `trace::start_decisions`/`finish_decisions` record every Metropolis
  `(weight, draw)` observationally (no RNG, numerical or state effect); `DecisionSummary` reports
  proposals, accepted, the smallest margin `|w - u|` and the near-flip count (margin below the
  1e-10 relative weight bound of the #424 harness). On the Heisenberg-chain fixture
  (121 254 proposals per walker, 200 samples) the smallest margin was 1.5e-7 to 3.2e-7, i.e.
  above a 1e-10 weight perturbation by three orders of magnitude, so a backend meeting the
  `accel_validation` bounds cannot flip a decision on this input.
* **Not covered.** Walker-parallel optimization (SR) needs the cross-walker reductions of the C
  group runs (`NSplitSize`); the runner covers the fixed-parameter sampling and measurement
  loop.

#### Device round-trip floor (measured)

RTX 3060, driver 580.178.04, CUDA driver API 13.0, toolkit 12.9, tenferro 0.7.1, Linux x86_64;
median of 30 after 5 warm-ups, pageable host memory, each step synchronized
(`cuda_gate_roundtrip_floor`, `scripts/run_cuda_gate.sh docker`).

| bytes | upload ms | upload GB/s | add+sync ms | download ms | download GB/s | total ms |
|---|---|---|---|---|---|---|
| 8 | 0.0704 | 0.00 | 0.0504 | 0.0715 | 0.00 | 0.1923 |
| 8192 | 0.1337 | 0.06 | 0.0901 | 0.1304 | 0.06 | 0.3542 |
| 65536 | 0.1782 | 0.37 | 0.0899 | 0.1369 | 0.48 | 0.4050 |
| 524288 | 0.6806 | 0.77 | 0.1019 | 0.2372 | 2.21 | 1.0196 |
| 4194304 | 3.8744 | 1.08 | 0.1582 | 1.0606 | 3.95 | 5.0932 |
| 33554432 | 73.7077 | 0.46 | 4.1402 | 25.0268 | 1.34 | 102.8748 |
| 134217728 | 295.5751 | 0.45 | 19.1972 | 102.9005 | 1.30 | 417.6728 |

The floor of one host-device round trip is about 0.19 ms (about 0.05 ms for a launch plus
synchronize, 0.07 ms each way for the smallest transfers). Upload saturates at about 1 GB/s
(0.45 GB/s beyond 32 MB), download at about 4 GB/s.

#### Break-even walker count

A walker-batched Pfaffian/inverse stage pays only if its cost, per step, beats the CPU doing
the same `W` walkers on `C` cores. The comparison below uses the measured single-thread CPU time
of the batched Pfaffian (`pfapack 1T`) and the CUDA kernel/total times of the batched engine of
#423 (`benchmark/gpu_pfaffian/results/pfaffian_batched.md` of PR #430, f64, `NQP = 8`, the
same host), expressed as the number of CPU cores the device replaces (`1T time of W walkers /
device time`):

| n | W | cores replaced, kernel only (planes already on the device) | cores replaced, with upload and download |
|---|---|---|---|
| 16 | 8 | 5.5 | 1.4 |
| 16 | 64 | 13.8 | 2.9 |
| 32 | 8 | 9.4 | 3.5 |
| 32 | 64 | 14.7 | 3.7 |
| 32 | 512 | 23.3 | 1.7 |
| 64 | 64 | 19.0 | 2.6 |
| 128 | 64 | 9.4 | 2.1 |

Reading: with transfers in every step the device replaces at most about 4 cores and below
about `W = 8` it replaces less than one (the 0.19 ms round-trip floor dominates the 0.05 to
0.4 ms CPU time of a single walker), so it never beats a walker-per-core CPU run on a host with
more than 4 cores. With the planes built and consumed on the device (kernel only) it replaces 14
to 24 cores for `W >= 64` at `n >= 32`, so the break-even is about `W = 8` against a single
core, about `W = 16` against 8 cores and unreachable against this 36-core host (a 3.4x to 7x
win over the all-core `rayon` column only at `W >= 64`, `n = 32`, and 1.4x to 3.6x for
`n >= 64`). RTX 3060 FP64 is 1/64 of FP32; a data-center GPU shifts these ratios. Numbers are
single-run on a shared host.

CPU walker scaling on the same host (`multichain_scaling_report`, Heisenberg-chain fixture,
200 samples per walker, one thread per walker, median of 5; ignored test, metadata block in its
output): the host was heavily loaded by other jobs (load average 58 to 81 on 36 threads), so
these are upper bounds on the cost of the runner itself, not a property of it.

| W | wall ms | ms per walker-chain | efficiency T(1)/T(W) |
|---|---|---|---|
| 1 | 1788.5 | 1788.5 | 1.00 |
| 2 | 1897.3 | 948.7 | 0.94 |
| 4 | 2126.0 | 531.5 | 0.84 |
| 8 | 3197.6 | 399.7 | 0.56 |
| 16 | 5761.5 | 360.1 | 0.31 |
| 32 | 15427.9 | 482.1 | 0.12 |

### 10.7 Host-device transfer diagnosis and the in-repo pinned/async helper (issue #432)

Measured in the standalone workspace (`gpu/mvmc-gpu-cuda`, ignored gate test
`transfer_microbenchmark_report`, `scripts/run_cuda_gate.sh docker`): RTX 3060 (sm_86, PCIe 4.0
x16, 28 SMs), driver 580.178.04, CUDA driver API 13.0, toolkit 12.9, cuBLAS 12.9.2, cuSOLVER
11.7.5, tenferro 0.7.1 (CubeCL CUDA), cudarc 0.19, Linux x86_64, single process, host
otherwise lightly loaded (load average about 2). **GPU copy engines: `asyncEngineCount = 2`**
(host-to-device and device-to-host copies can run concurrently with each other and with a
kernel), `concurrentKernels = 1`. Medians over 5 to 60 repetitions (fewer for large sizes);
three runs of the sweep agreed to within a few percent except where noted.

#### Per-call time and bandwidth versus size

(1) cudarc pinned memcpy, (2) cudarc pageable memcpy, (3) tenferro `upload_tensor` /
`download_tensor` (`CudaBackend` runtime, F64 vector, synchronized). Sizes of the planes
`[n, n, NQP = 8, B]` of #423: n = 16, B = 8: 131 KB; n = 32, B = 64: 4.2 MB; n = 64, B = 64:
16.8 MB; n = 128, B = 64: 67 MB. Per-walker vectors are 4 KB to 1 MB.

| bytes | pageable up ms (GB/s) | pageable down ms (GB/s) | pinned WC up ms (GB/s) | pinned cached up ms (GB/s) | pinned down ms (GB/s) | pinned up enqueue ms | tenferro up ms (GB/s) | tenferro down ms (GB/s) |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.0056 (0.00) | 0.0068 (0.00) | 0.0092 (0.00) | 0.0061 (0.00) | 0.0071 (0.00) | 0.0038 | 0.0214 (0.00) | 0.0264 (0.00) |
| 512 | 0.0073 (0.07) | 0.0075 (0.07) | 0.0153 (0.03) | 0.0065 (0.08) | 0.0072 (0.07) | 0.0099 | 0.0227 (0.02) | 0.0269 (0.02) |
| 4096 | 0.0088 (0.47) | 0.0083 (0.50) | 0.0521 (0.08) | 0.0071 (0.58) | 0.0075 (0.54) | 0.0466 | 0.0316 (0.13) | 0.0278 (0.15) |
| 32768 | 0.0184 (1.78) | 0.0141 (2.33) | 0.0104 (3.14) | 0.0092 (3.57) | 0.0096 (3.40) | 0.0035 | 0.0389 (0.84) | 0.0295 (1.11) |
| 131072 | 0.0397 (3.30) | 0.0331 (3.96) | 0.0193 (6.79) | 0.0170 (7.70) | 0.0173 (7.58) | 0.0036 | 0.0864 (1.52) | 0.0468 (2.80) |
| 1048576 | 0.2339 (4.48) | 0.2217 (4.73) | 0.0947 (11.07) | 0.0916 (11.44) | 0.0885 (11.85) | 0.0043 | 0.6585 (1.59) | 0.2117 (4.95) |
| 4194304 | 0.6848 (6.13) | 0.5812 (7.22) | 0.3486 (12.03) | 0.3442 (12.18) | 0.3338 (12.56) | 0.0043 | 5.8194 (0.72) | 4.2021 (1.00) |
| 16777216 | 2.4114 (6.96) | 2.1493 (7.81) | 1.3626 (12.31) | 1.3571 (12.36) | 1.3171 (12.74) | 0.0043 | 18.4228 (0.91) | 5.0354 (3.33) |
| 67108864 | 8.6923 (7.72) | 6.6501 (10.09) | 5.4070 (12.41) | 5.3975 (12.43) | 5.0960 (13.17) | 0.0065 | 130.0937 (0.52) | 49.2461 (1.36) |
| 268435456 | 34.4588 (7.79) | 22.2363 (12.07) | 21.5796 (12.44) | 21.5610 (12.45) | 20.3493 (13.19) | 0.0064 | 509.0499 (0.53) | 187.1178 (1.43) |

"pinned WC" is write-combined pinned memory (cudarc's default `alloc_pinned`), "pinned cached"
ordinary pinned memory; the download column is cached pinned memory; "enqueue" is the host time
of one pinned copy call that does not wait.

#### Cause of the slow tenferro transfers

* The PCIe link and the driver are not the limit: pageable cudarc copies reach 7 to 8 GB/s
  up and 8 to 12 GB/s down for 16 to 256 MB, pinned 12.4 to 13.2 GB/s (about the practical
  PCIe 4.0 x16 limit).
* tenferro 0.7.1 reaches only 0.5 to 1.7 GB/s up and 1.0 to 5 GB/s down (the 0.5 and 0.8 GB/s
  of the #423 report): 1.5 to 15 times slower than plain pageable cudarc at 1 MB and larger
  (15x up and 8x down at 256 MB), and its per-call fixed cost is 19 to 26 us against 5 to 9 us
  for cudarc (about 3.5 times).
* Missing pinned memory is therefore not the main cause: pageable cudarc is already faster than
  tenferro by the factors above. The code path (`upload_tensor`: CubeCL `create_from_slice`;
  `download_tensor`: `rt.synchronize()`, CubeCL `read_one`, then a second host copy
  `T::from_bytes(..).to_vec()`; `third_party/tenferro-gpu/src/cubecl/memory.rs`) adds a device
  synchronization, CubeCL staging and extra host copies. Instrumenting the sources with timers
  attributes the cost (256 MB, after warm-up): an upload spends 181 ms in
  `slice.to_vec()` (`t4a-cubecl-runtime` `client.rs:296`) and 381 ms in a second
  `data.to_vec()` (`client.rs:232`) against 96 ms for the actual pageable host-to-device copy
  (87 % redundant host copies; 68 % at 16 MB); a download spends 20.5 ms in `read_one` and
  162 ms in `from_bytes(..).to_vec()` (`memory.rs:222`; 82 %, 61 % at 16 MB). Above 100 MB the
  CubeCL host pool is bypassed for `vec![0; n]` (`t4a-cubecl-cuda` `command.rs:162`). The
  copies run at 0.7 to 1.4 GB/s because every call writes into new, page-faulting memory. Full
  analysis with file:line references and the minimal example are in
  `docs/design/tenferro-transfer-request-draft.md`.
* Write-combined memory is a trap for small copies: a 4 KB host-to-device copy takes 51 us from
  write-combined pinned memory against 7 us from cached pinned memory (reproduced in every run),
  and 11 us for 32 KB; the bandwidth is the same for 1 MB and larger. `PinnedKind::for_upload`
  therefore uses cached pinned memory below 32 KB.
* For small per-step transfers (up to 64 KB, the walker configuration and pf/energy vectors)
  pageable cudarc is within 2 to 3 us of pinned; pinned memory pays for large copies
  (16 MB: 1.8x up and 1.6x down; 256 MB: 1.6x up and 1.1x down) and, more importantly,
  makes the copy asynchronous with respect to the host: enqueueing a 16 MB pinned copy takes
  about 4 us while the pageable call blocks the host for 2.3 ms.

#### Overlap with a concurrent kernel

copy size 16777216 bytes, kernel calibrated to 1.368 ms

| scenario | copy ms | kernel ms | both ms | overlap |
|---|---|---|---|---|
| pinned host-to-device + kernel | 1.365 | 1.368 | 1.371 | 1.00 |
| pinned device-to-host + kernel | 1.279 | 1.369 | 1.370 | 1.00 |
| pinned both directions + kernel | 1.762 | 1.368 | 1.767 | 1.00 |
| pageable host-to-device (on a second stream) + kernel | 1.458 | 1.370 | 1.464 | 1.00 |
| pageable device-to-host (on a second stream) + kernel | 1.368 | 1.369 | 1.375 | 1.00 |

(copy and kernel on different streams; overlap = (copy + kernel - both) / min(copy, kernel),
1.00 = the shorter one is completely hidden.) Pinned copies hide completely behind a kernel and
both directions run concurrently with it (two copy engines); pageable cudarc copies on a
separate stream also overlap on the device, but block the calling host thread for their whole
duration, so only pinned copies free the host to decide the next step while the device works.

#### Ping-pong of two walker groups

Group A and group B on two streams: while the device runs group A's kernel and copies, the host
decides group B (a busy wait standing for the Metropolis decision and SFMT draws), then they
swap. Baselines run the same 2 x steps cycles on one stream with a full wait every step, with
pageable synchronous copies ("current practice") and with pinned asynchronous copies.

| in bytes | out bytes | kernel ms | host us | steps/group | serial pageable ms | serial pinned ms | ping-pong ms | speedup vs pageable | speedup vs pinned serial |
|---|---|---|---|---|---|---|---|---|---|
| 4096 | 4096 | 0.05 | 20 | 400 | 66.1 | 68.4 | 41.6 | 1.59x | 1.64x |
| 4096 | 65536 | 0.30 | 100 | 200 | 170.1 | 169.8 | 120.0 | 1.42x | 1.41x |
| 4096 | 65536 | 1.00 | 300 | 200 | 530.2 | 529.4 | 399.5 | 1.33x | 1.33x |
| 65536 | 1048576 | 1.00 | 200 | 100 | 277.7 | 262.1 | 199.9 | 1.39x | 1.31x |

Reading: two groups hide 25 to 40 % of the per-step time (the ideal for two groups when host
and device time are equal is 2x); the benefit comes from the pipeline, not from pinning alone
(the two serial baselines are equal within noise because the host decision dominates). The gain
grows with the number of groups in flight; #434 should use more than two streams or groups when
the host decision time is shorter than the device time.

#### Decision and helper

tenferro's transfer path is the bottleneck, not PCIe and not pinned memory. The hot transfers of
the device-resident sampler (#434) should therefore bypass `upload_tensor` / `download_tensor`:
`gpu/mvmc-gpu-cuda/src/transfer.rs` provides

* `PinnedBuf` / `PinnedPool` (cached or write-combined pinned memory, size-class reuse; pinned
  allocation costs about a millisecond and must not happen per step),
* `TransferStream::{upload_async, download_async}` returning a `Pending` that borrows the host
  buffer until the copy completed (the borrow checker prevents touching a buffer the device may
  still use; dropping a `Pending` waits) and `record` / `wait_event` / `wait_pending` for
  device-side cross-stream dependencies (no device-wide synchronize),
* `unsafe upload_raw` / `download_raw` for device memory owned by another allocator (for
  example a tenferro raw session address; not exercised by the gate),
* `disable_event_tracking`: cudarc's implicit event tracking must be off so the safe wrappers do
  not add hidden synchronization; the helper orders everything with explicit events.

The gate test `transfer_helper_is_correct_and_orders_streams` checks data integrity, pool reuse
and event ordering on the device. Large plane uploads should still be avoided altogether by
building the planes on the device (#425, #426): even at pinned speed 67 MB takes 5.4 ms.

An upstream request draft with these numbers is in
`docs/design/tenferro-transfer-request-draft.md` (not filed; filing needs approval).

### 10.8 Multi-walker optimization with C-compatible reductions (issue #435)

`mvmc_core::multichain::run_para_opt_multichain` extends the #425 runner from fixed-parameter
PhysCal to the whole optimization (ParaOpt, `NSROptItrStep` SR steps). C runs `W` independent
chains as the `W` ranks of an ungrouped run (`NSplitSize = 1`): walker `w` samples its own chain
(seed `RndSeed + w`), `HO`, `OO`, the energy and the weights are summed over all ranks with
`MPI_Allreduce(MPI_SUM)`, every rank solves the same SR system and applies the same update,
and `SROptO` stays rank-local. The Rust optimizer is already written against the `Reducer`
trait, so the multi-walker run is the unchanged optimizer with `ThreadReducer`, an in-process
reducer in which `W` walker threads stand in for the MPI communicator (sums, broadcasts, MAX of
the comm1 INFO, barrier, `rank`/`world_size`/`seed_offset`). No SR code is duplicated, and the
collective sequence, seeds and draw order are those of the MPI run; the SR assembly can run
through either backend of #421 (`MVMC_RS_SR_BACKEND`).

* **Reduction order.** `ThreadReducer` sums in rank order as a left fold,
  `((v0 + v1) + v2) + ...`, identical on every walker and independent of thread scheduling.
  `MPI_Allreduce` leaves the order to the MPI library, so Rust and C can differ by last-bit
  roundoff for `W > 2` (`W <= 2` is a single sum of two terms; observed agreement is within the
  1e-13 + 1e-12 relative bound of `tests/fixtures/mpi_matrix_179/README.md` in every cell).
* **Threads.** The collectives need all walkers running, so one OS thread per walker is used
  (not a pool smaller than `W`); everything inside a walker is single-threaded. The SR tenferro
  backend is a process-wide, mutex-guarded runtime (one runtime built once), so walkers
  serialize on it during an SR solve; no collective is issued while it is held.
* **Not covered.** Intra-group splitting (`NSplitSize > 1`: QP/sample split of one chain over
  several ranks) is not a walker; the `W = ranks / NSplitSize` groups of such a C run compute the
  same chains and operands (test below) to reduction-order roundoff. Output files are written by
  walker 0 only, as the output root of the MPI run.

Validation (`crates/mvmc-core/tests/multiwalker_sr_435.rs`, no MPI, C or Julia needed):

* every ungrouped C cell of the #179 matrix with 2 and 4 ranks and width 1 (models real, cmp,
  FSZ with `NQPFull` 1 and 2, OptTrans; solvers direct `NStore` 0 and 1 and SR-CG; including
  the uneven-work cells) is reproduced per walker: `Counter[0..6]`, every saved `EleIdx` and the
  full SFMT state exactly, the reduced `<HO>`, `<OO>`, `<O>` (walker 0) and the step energy
  within the bound above (OptTrans real-mode derivative slots excluded as in the MPI gate,
  #370);
* two walkers equal the two groups of the 4-rank, `NSplitSize = 2` C cells (chains with seeds 0
  and 1: group leaders' configurations and RNG states exactly, reduced operands within the bound);
* `W = 1` is byte-identical to the serial optimization (output files except wall-clock timing
  files, the whole final state and the RNG) over four steps and 40 samples for real, cmp, fsz2
  and ot;
* beyond step 1 (ill-conditioned SR, #358) a four-step 4-walker run is bitwise repeatable and
  all walkers agree on the reduced energy; solver trajectories are not forced onto C;
* the same checks hold with the SR assembly on the tenferro backend (sampling and the reduced
  operands do not depend on the solver at step 1; `W = 1` equals the serial tenferro run);
* `ThreadReducer` collectives follow the MPI contract (rank-ordered sum, broadcast from the
  root, MAX, seed offsets).

## 11. Batched Pfaffian and inverse (issue #423)

Status: implemented, validated and benchmarked on hardware (2x RTX 3060). Not wired into the
production sampler (that needs #422 and #425). Related to #417 and #420.

### 11.1 API and layout

`mvmc_gpu::pfaffian_inverse_batched(&Backend, planes, n, nqp, batch)` takes a column-major
`[n, n, NQP, B]` batch of skew-symmetric `f64` or `Complex64` planes (plane `p = q + NQP*b`,
`n` even) and returns `pf [NQP, B]`, `inv [n, n, NQP, B]` and a per-plane `PlaneStatus`
(`Ok`, `ZeroPivot { row }` with pfapack's 1-based `INFO`, `NonFinite`). `inv` is the true
`A^-1` (`utu2inv`); the production sampler's final sign flip stays with the caller. For a failed
plane the inverse is zero-filled and a zero-pivot plane reports `pf = 0`. There is no silent
fallback: an unavailable backend is a typed `Error`.

| backend | where | what |
| --- | --- | --- |
| `CpuPfapack`, `CpuPfapackRayon` | `crates/mvmc-gpu` (main workspace, normal CI) | pfapack per plane (`dsktf2`/`zsktf2`, `utu2pfa`, `utu2inv`), one workspace per task; bit-identical to calling pfapack |
| `TenferroExtension` | `crates/mvmc-gpu` | the same per-plane kernel as a tenferro `ExtensionOp` (`define_extension_runtime!`, `execute_reads`, zero-copy `as_slice` input, per-`(dtype, n)` workspace in the runtime's `ExtensionCacheStore`) run in a tenferro `Runtime` on `CpuBackend`; bit-identical to pfapack |
| `TenferroNative` | `crates/mvmc-gpu` | batched Parlett-Reid / LTL^T written with tenferro tensors only (11.5) |
| `Engine(&CudaEngine)` | `gpu/mvmc-gpu-cuda` (`pfaffian.rs`, `pfaffian_batched.cu`) | one CUDA thread block per plane, NVRTC kernel launched through tenferro's raw session |

`mvmc-gpu` has no CUDA dependency; the CUDA engine implements its `BatchedEngine` trait and
plugs in through `Backend::Engine`. The root `Cargo.lock` gains only the `mvmc-gpu` package
entry (no new external dependency). `gpu/mvmc-gpu-cuda` (own lock, #420 layout) depends on
`mvmc-gpu`, reuses the #420 gate decision (`cuda_gate_decision`, `CUDA_GATE_VARIABLE`) and the
device report, and is exercised by `scripts/run_cuda_gate.sh`.

### 11.2 CUDA kernel design

Same math as pfapack, ported thread-parallel: one block per plane (32 to 256 threads), the plane
in global memory, `O(n)` vectors in dynamic shared memory, `n <= 1024`.

1. LTL^T, `k0 = n-1 .. 1`: first-index argmax of `|A[0..kk, k0]|` (`|re|+|im|` for complex, as
   `izamax`; NaN loses except at index 0, as in the scalar `v > colmax` scan) by a block tree
   reduction; the row/column swap with the upper-triangle sign twiddles of `dsktf2` (all swapped
   elements are disjoint, one barrier between swap and negation); the skew rank-2 update
   `A_ij = (A_ij + x_i t1_j) - y_i t2_j` over the strict upper triangle with threads striding
   `i + kk*j`; then the column scaling. A zero column records `INFO` once (the first met,
   scanning from `n`) and skips the step, like pfapack's `continue`.
2. Pfaffian: sequential product of `A[i, i+1]` and the pivot sign by thread 0, finiteness test.
3. Inverse (`utu2inv`): one thread per column for the unit upper-triangular inverse (back
   substitution into a workspace plane) and for the skew-tridiagonal solve, the composed pivot
   permutation built once in shared memory, and the final `M^T C` product with both
   permutations folded into the index map (one thread per output element).

NVRTC runs with `--fmad=false`, so the rank-2 update keeps pfapack's operation order. The
matrix stays in global memory, not shared memory: a complex `n = 128` plane is 256 KB (real
128 KB), above the 99 KB opt-in shared limit of sm_86, so the working set is served by L1/L2.
A shared-memory variant for small `n` is a possible later optimisation, not needed for the
numbers below. Planes are processed in chunks of at most 3 GiB of device buffers (work copy,
workspace, output).

tenferro raw API notes (0.7.1): `with_raw`, `compile_nvrtc`, `launch`, `upload_bytes`,
`alloc_output` and `download_tensor` are sufficient (no `cudarc` needed). `raw::Module` is
`!Send`, so it cannot live in the `Send`-only `Session::resource` cache; the module is compiled
once per `with_session` scope (tens of milliseconds, the driver caches the PTX JIT), and the
`CudaEngine` convenience wrapper recompiles on every call. `alloc_output` memory is
uninitialised (the kernel writes every output element). A kernel that updates its input in
place needs `DeviceBytes` (`upload_bytes` + `KernelArg::workspace`) because a `TensorRef` is
read-only. There are no pinned or asynchronous transfers (rates in 11.4).

### 11.3 Accuracy

Gate `gpu/mvmc-gpu-cuda/tests/pfaffian_gate.rs` (`MVMC_RS_CUDA_GATE=1
scripts/run_cuda_gate.sh docker`; 32 random planes per size, NQP = 8, B = 4; RTX 3060), GPU
against `CpuPfapack`. `observed/allowed` is the worst of the inverse and Pfaffian relative errors
divided by `16 * n * eps * cond_F(A)` (`cond_F = |A|_F |A^-1|_F` of that plane), the standard
forward-error scale of a backward-stable inverse. The kernel uses pfapack's pivot rule and, with
`--fmad=false`, its real operation order, so the factor `16` is generous.

| dtype | n | max inverse rel. err | max Pf rel. err | max cond_F | observed / allowed |
| --- | ---: | ---: | ---: | ---: | ---: |
| f64 | 2 / 4 | 0 / 0 | 0 / 0 | 2e0 / 1.9e2 | 0 |
| f64 | 6 | 1.8e-16 | 0 (bit-identical) | 2.2e2 | 1e-3 |
| f64 | 16 | 8.9e-16 | 0 | 2.1e3 | < 1e-3 |
| f64 | 32 | 1.8e-15 | 0 | 4.7e4 | < 1e-3 |
| f64 | 64 | 3.8e-15 | 0 | 1.4e4 | < 1e-3 |
| f64 | 128 | 7.7e-15 | 0 | 8.3e4 | < 1e-3 |
| c64 | 2 | 2.0e-16 | 0 | 2e0 | 1.4e-2 |
| c64 | 4 / 6 | 1.7e-15 / 9.6e-16 | 1.7e-15 / 1.0e-15 | 2.1e2 / 9.9e1 | 4e-3 |
| c64 | 16 | 1.1e-14 | 1.2e-14 | 6.9e2 | 1e-3 |
| c64 | 32 | 8.6e-15 | 7.4e-15 | 6.1e2 | < 1e-3 |
| c64 | 64 | 6.3e-14 | 6.6e-14 | 4.4e3 | < 1e-3 |
| c64 | 128 | 3.3e-13 | 3.3e-13 | 1.9e4 | < 1e-3 |

The real Pfaffian is bit-identical to pfapack at every size (same pivots, same update order);
the real inverse differs only by the order of the dot products in the triangular inverse and in
the final product. The complex results differ through the division convention (pfapack uses
Julia's division, the kernel Smith's algorithm) and complex multiply association; they still
sit more than two orders of magnitude below the conditioning scale. Independent invariants of
the GPU result (not relative to the CPU result): `max|A A^-1 - I|`, the relative skew defect of
`A^-1` and `Pf^2 = det A` (dense LU on the CPU, sign included) are each below the same
`16 n eps cond` scale (`4x` for `Pf^2`). Zero-pivot planes (all zero; a zero row/column) report
`ZeroPivot`, a NaN plane a non-`Ok` status, neighbouring planes are unaffected, and CPU and GPU
statuses agree exactly. The tensor-native and `ExtensionOp` backends also agree with the kernel
(`1e-12` Pfaffian, `1e-11` inverse, `n = 16`).

Normal CI (no GPU) covers: CPU batched and `ExtensionOp` bit-identical to pfapack (real and
complex, `n = 2..32`); `Pf^2 = det`, `A A^-1 = I` and skew structure of every CPU backend; the
tensor-native backend against pfapack (real Pfaffian bit-identical, inverse relative error
`7e-18..1.4e-15`, complex `2e-16..6e-15`); statuses; shape errors.

### 11.4 Benchmark

Environment: Intel Xeon E5-2699 v3 (36 hardware threads, shared host: load average 4.6 at the start and 18.8 at the end because other jobs were running), 2x NVIDIA GeForce RTX 3060 (sm_86, 12 GB, benchmark on device 0), NVIDIA driver 580.178.04, CUDA driver API 13.0, container CUDA toolkit 12.9.2 (nvcc 12.9.86, NVRTC 12.9; image `tenferro-benchmark-cuda:full-verify-20260822`), rustc 1.98.0 (container), tenferro 0.7.1, Linux x86_64. Medians of up to 7 runs after one warm-up (stop after 12 s, at least 3 runs), times in milliseconds, NQP = 8, planes = 8 B. CPU rows run `pfapack` with its default scalar features (no BLAS, no SIMD). CSV: `benchmark/gpu_pfaffian/results/pfaffian_batched.csv`; reproduce with `scripts/run_pfaffian_bench.sh docker`.

`CUDA total` is the call as a caller sees it (upload, allocation, kernel, download, host glue).
Tensor-native and `ExtensionOp` are skipped when a single run would take minutes (`-`). The host
was shared with other jobs while measuring, so CPU rows, rayon in particular, carry noise: a few
single-thread outliers are visible (for example f64 `n = 32, B = 8`, c64 `n = 64, B = 1` and
`B = 8`), and the CPU rows use scalar pfapack (the production `calc_m_all` uses the BLAS-backed
path measured at 135 us per `n = 64` plane on an unloaded host in section 3.2, about half of the
267 us per plane of the one-thread row here), so speedups against the production CPU path are
roughly 2x lower than the ratios below. Times in milliseconds.

| dtype | n | B | planes | pfapack 1T | pfapack rayon | tenferro native | tenferro ExtOp | CUDA total | CUDA kernel | up / alloc / down | f32 kernel | kernel vs 1T | kernel vs rayon | total vs 1T | total vs rayon | f64/f32 kernel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f64 | 16 | 1 | 8 | 0.05 | 0.13 | 20.82 | 8.93 | 0.38 | 0.13 | 0.05 / 0.05 / 0.14 | 0.08 | 0.4x | 1.0x | 0.1x | 0.3x | 1.6x |
| f64 | 16 | 8 | 64 | 0.77 | 0.33 | 34.07 | 8.30 | 0.55 | 0.14 | 0.14 / 0.05 / 0.20 | 0.10 | 5.5x | 2.3x | 1.4x | 0.6x | 1.4x |
| f64 | 16 | 64 | 512 | 5.68 | 2.71 | 172 | 11.68 | 1.93 | 0.41 | 0.72 / 0.06 / 0.47 | 0.15 | 13.8x | 6.6x | 2.9x | 1.4x | 2.7x |
| f64 | 16 | 512 | 4096 | 22.25 | 13.16 | 566 | 57.99 | 40.71 | 2.61 | 20.37 / 2.59 / 13.14 | 0.62 | 8.5x | 5.0x | 0.5x | 0.3x | 4.2x |
| f64 | 16 | 4096 | 32768 | 239 | 129 | - | 325 | 295 | 20.21 | 137 / 9.44 / 82.28 | 4.35 | 11.8x | 6.4x | 0.8x | 0.4x | 4.6x |
| f64 | 32 | 1 | 8 | 0.42 | 0.39 | 55.34 | 16.84 | 0.62 | 0.33 | 0.10 / 0.04 / 0.15 | 0.21 | 1.3x | 1.2x | 0.7x | 0.6x | 1.5x |
| f64 | 32 | 8 | 64 | 3.42 | 3.91 | 342 | 21.39 | 0.98 | 0.36 | 0.28 / 0.03 / 0.23 | 0.21 | 9.4x | 10.8x | 3.5x | 4.0x | 1.7x |
| f64 | 32 | 64 | 512 | 27.39 | 6.37 | 1877 | 52.04 | 7.48 | 1.87 | 2.70 / 0.05 / 1.73 | 0.59 | 14.7x | 3.4x | 3.7x | 0.9x | 3.2x |
| f64 | 32 | 512 | 4096 | 266 | 82.71 | - | 351 | 152 | 11.43 | 70.88 / 5.57 / 41.68 | 3.23 | 23.3x | 7.2x | 1.7x | 0.5x | 3.5x |
| f64 | 32 | 4096 | 32768 | 2046 | 532 | - | 2422 | 1130 | 83.91 | 526 / 31.81 / 313 | 24.36 | 24.4x | 6.3x | 1.8x | 0.5x | 3.4x |
| f64 | 64 | 1 | 8 | 2.09 | 0.53 | 98.09 | 17.49 | 1.53 | 0.99 | 0.20 / 0.04 / 0.22 | 0.61 | 2.1x | 0.5x | 1.4x | 0.3x | 1.6x |
| f64 | 64 | 8 | 64 | 16.70 | 2.52 | 568 | 32.41 | 4.23 | 1.54 | 1.25 / 0.05 / 0.89 | 0.79 | 10.8x | 1.6x | 4.0x | 0.6x | 1.9x |
| f64 | 64 | 64 | 512 | 179 | 17.79 | - | 216 | 69.47 | 9.45 | 30.46 / 7.12 / 18.97 | 6.31 | 19.0x | 1.9x | 2.6x | 0.3x | 1.5x |
| f64 | 64 | 512 | 4096 | 1156 | 226 | - | 1450 | 577 | 62.27 | 258 / 20.56 / 151 | 40.03 | 18.6x | 3.6x | 2.0x | 0.4x | 1.6x |
| f64 | 64 | 4096 | 32768 | 8750 | 1348 | - | - | 4557 | 486 | 2049 / 64.89 / 1295 | 304 | 18.0x | 2.8x | 1.9x | 0.3x | 1.6x |
| f64 | 128 | 1 | 8 | 8.97 | 3.61 | 642 | 25.73 | 6.13 | 4.01 | 0.98 / 0.06 / 0.71 | 2.10 | 2.2x | 0.9x | 1.5x | 0.6x | 1.9x |
| f64 | 128 | 8 | 64 | 90.28 | 15.22 | - | 108 | 39.77 | 10.43 | 19.91 / 0.05 / 7.35 | 4.56 | 8.7x | 1.5x | 2.3x | 0.4x | 2.3x |
| f64 | 128 | 64 | 512 | 708 | 187 | - | 818 | 339 | 75.32 | 133 / 10.54 / 77.13 | 39.78 | 9.4x | 2.5x | 2.1x | 0.6x | 1.9x |
| f64 | 128 | 512 | 4096 | 5397 | 1202 | - | - | 2643 | 580 | 1034 / 39.23 / 631 | 276 | 9.3x | 2.1x | 2.0x | 0.5x | 2.1x |
| f64 | 128 | 4096 | 32768 | 42805 | 8801 | - | - | 22801 | 4631 | 9359 / 386 / 5264 | 2180 | 9.2x | 1.9x | 1.9x | 0.4x | 2.1x |
| c64 | 16 | 1 | 8 | 0.10 | 0.16 | 20.85 | 8.58 | 0.37 | 0.17 | 0.05 / 0.03 / 0.10 | 0.09 | 0.6x | 0.9x | 0.3x | 0.4x | 2.0x |
| c64 | 16 | 8 | 64 | 0.72 | 0.31 | 43.07 | 8.77 | 0.90 | 0.24 | 0.27 / 0.06 / 0.25 | 0.11 | 3.0x | 1.3x | 0.8x | 0.3x | 2.2x |
| c64 | 16 | 64 | 512 | 7.33 | 3.07 | 192 | 14.33 | 3.86 | 1.09 | 1.36 / 0.05 / 0.87 | 0.20 | 6.7x | 2.8x | 1.9x | 0.8x | 5.6x |
| c64 | 16 | 512 | 4096 | 48.68 | 17.31 | 1047 | 99.71 | 98.39 | 7.84 | 47.45 / 6.62 / 30.68 | 0.98 | 6.2x | 2.2x | 0.5x | 0.2x | 8.0x |
| c64 | 16 | 4096 | 32768 | 368 | 175 | - | 547 | 602 | 57.82 | 273 / 20.85 / 160 | 6.84 | 6.4x | 3.0x | 0.6x | 0.3x | 8.4x |
| c64 | 32 | 1 | 8 | 0.96 | 0.32 | 43.92 | 16.19 | 1.02 | 0.60 | 0.11 / 0.03 / 0.27 | 0.27 | 1.6x | 0.5x | 0.9x | 0.3x | 2.2x |
| c64 | 32 | 8 | 64 | 7.79 | 2.11 | 217 | 22.89 | 2.98 | 0.93 | 0.95 / 0.07 / 0.64 | 0.32 | 8.4x | 2.3x | 2.6x | 0.7x | 2.9x |
| c64 | 32 | 64 | 512 | 75.68 | 12.02 | 989 | 82.74 | 37.50 | 5.36 | 21.30 / 0.06 / 8.47 | 1.09 | 14.1x | 2.2x | 2.0x | 0.3x | 4.9x |
| c64 | 32 | 512 | 4096 | 579 | 145 | - | 685 | 295 | 39.20 | 129 / 10.65 / 74.71 | 6.32 | 14.8x | 3.7x | 2.0x | 0.5x | 6.2x |
| c64 | 32 | 4096 | 32768 | 4372 | 736 | - | 5096 | 3168 | 302 | 1179 / 59.42 / 884 | 48.65 | 14.5x | 2.4x | 1.4x | 0.2x | 6.2x |
| c64 | 64 | 1 | 8 | 24.98 | 5.98 | 554 | 39.75 | 3.07 | 2.31 | 0.34 / 0.04 / 0.29 | 0.88 | 10.8x | 2.6x | 8.1x | 1.9x | 2.6x |
| c64 | 64 | 8 | 64 | 163 | 9.32 | 3024 | 59.00 | 11.05 | 5.41 | 2.64 / 0.06 / 1.74 | 1.31 | 30.2x | 1.7x | 14.8x | 0.8x | 4.1x |
| c64 | 64 | 64 | 512 | 381 | 88.08 | - | 413 | 194 | 34.91 | 79.74 / 5.57 / 46.88 | 9.96 | 10.9x | 2.5x | 2.0x | 0.5x | 3.5x |
| c64 | 64 | 512 | 4096 | 3095 | 468 | - | 3379 | 1490 | 244 | 626 / 30.73 / 374 | 68.48 | 12.7x | 1.9x | 2.1x | 0.3x | 3.6x |
| c64 | 64 | 4096 | 32768 | 24328 | 2677 | - | - | 11115 | 1928 | 4576 / 198 / 2767 | 548 | 12.6x | 1.4x | 2.2x | 0.2x | 3.5x |
| c64 | 128 | 1 | 8 | 31.10 | 7.24 | 1137 | 46.84 | 15.71 | 13.26 | 1.20 / 0.04 / 0.77 | 1.51 | 2.3x | 0.5x | 2.0x | 0.5x | 8.8x |
| c64 | 128 | 8 | 64 | 256 | 29.12 | - | 334 | 97.73 | 36.84 | 32.71 / 3.87 / 20.88 | 3.01 | 7.0x | 0.8x | 2.6x | 0.3x | 12.3x |
| c64 | 128 | 64 | 512 | 2088 | 337 | - | 2417 | 728 | 215 | 257 / 21.14 / 150 | 17.86 | 9.7x | 1.6x | 2.9x | 0.5x | 12.1x |
| c64 | 128 | 512 | 4096 | 16670 | 2278 | - | - | 5795 | 1666 | 2063 / 96.95 / 1238 | 134 | 10.0x | 1.4x | 2.9x | 0.4x | 12.4x |
| c64 | 128 | 4096 | 32768 | skipped (> 4 GiB of planes) | | | | | | | | | | | | |


What the table says (FP64 / C64, the physics path):

* **Kernel only, data resident on the device** (the design of #422/#426, where planes are built
  and consumed on the device): for `B >= 8` the kernel is 5.5 to 14x (f64 `n = 16`), 9 to 24x
  (`n = 32`), 11 to 19x (`n = 64`) and 9x (`n = 128`) faster than one pfapack thread (c64: 3 to
  7x, 8 to 15x, 11 to 30x, 7 to 10x), and 0.8 to 11x faster than all 36 threads (at least 1.3x
  except c64 `n = 128, B = 8`; f64 `n = 32`: 3.4 to 11x, `n = 64`: 1.6 to 3.6x, `n = 128`: 1.5 to
  2.5x). At `B = 1` (8 planes) the launch is not worth it: 0.4x of one thread at `n = 16`, 1.3x
  at `n = 32`.
* **Including upload and download** (an offload-only design): 1.4 to 4x faster than one thread
  for `n >= 32` and `B >= 8` (15x on one noisy c64 row), but slower than all 36 threads in almost
  every cell (0.2 to 0.9x); the GPU wins only in a few small cells (f64 `n = 16, B = 64` 1.4x,
  f64 `n = 32, B = 8` 4.0x against a noisy rayon row, c64 `n = 64, B = 1` 1.9x). For `n = 16`
  with `B >= 512` the transfers take 10 to 13x the kernel and the GPU is slower than even one
  thread. The cause is the transfer path: tenferro's `upload_bytes` / `download_tensor` move
  pageable memory at about 0.5 GB/s up (1.07 GB in 2.05 s, f64 `n = 64, B = 4096`) and 0.8 GB/s
  down, far below PCIe 3.0 x16 (about 12 GB/s). The conclusion: do not offload the Pfaffian
  plane by plane through host memory; it pays when the planes are built and consumed on the
  device (or with pinned asynchronous transfers, which tenferro 0.7.1 does not offer).
* **Crossover (data resident):** the GPU beats one thread from 64 planes (`B = 8`) at every
  `n` (already at 8 planes for `n >= 32`), and beats 36 threads by 2x or more for `n <= 32`
  from `B = 8`, by 1.5 to 3.6x for `n = 64` and `n = 128`.
* **FP64 penalty:** FP64 runs at 1/64 of FP32 on this card, but the single-precision kernel
  (timing only) is only 1.4 to 4.6x faster for f64 and 2 to 12.4x for c64: the kernel is limited
  by global-memory traffic and its serial phases (argmax reduction, `n - 1` dependent LTL steps,
  the column recurrences of the tridiagonal solve), not by FLOPs. The ratio grows with the
  arithmetic per byte (c64 `n = 128`: 12x). A data-centre GPU (FP64 at 1/2 of FP32) would be
  close to the f32 column; on this card the f64 kernel is already 9 to 24x faster than one core.
* **Production scale:** the Hubbard inputs have `n = 16..64` and 8 QP planes per sample; 64
  batch elements (512 planes) take 0.41 / 1.87 / 9.45 ms on the kernel (f64, `n = 16 / 32 / 64`)
  against 5.7 / 27 / 179 ms for one thread (14 to 19x). The 58 to 61 % of wall time spent in
  `CalculateMAll` and the recalculation (section 3.2) could therefore shrink by an order of
  magnitude if the planes stay on the device, before adding the transfers and the rest of the
  sampler (Amdahl bounds of section 6).

### 11.5 tenferro-native CPU reference and its gaps

`TenferroNative` shows what a device-portable tenferro formulation looks like: all state in
tensors with the plane axis trailing, no per-plane host loop, one session per call. It uses
pfapack's pivot rule and elementwise update order (so the real Pfaffian is bit-identical) and
replaces the sequential kernels by whole-batch operations: the pivot search by `reduce_max` +
`compare` + `select` + `reduce_min`, the batch-varying row/column swap by one-hot `select`
masks and `reduce_sum`, the skew rank-2 update by broadcast elementwise operations with the
block re-skewed through `transpose` and `select`, the unit upper-triangular inverse by
batched back substitution over rows (broadcast multiply + `reduce_sum`), the skew-tridiagonal solve by row
recurrences over `[n, P]` slices, and the pivot permutations by batched one-hot permutation
matrices applied with `dot_general`.

Stability (issue #466). The first version expanded the unit upper-triangular inverse as
`(I+N)^-1 = prod_j (I + (-N)^(2^j))` (repeated squaring of the multiplier matrix). That is exact
in exact arithmetic but numerically unstable: the entries of the powers of `N` grow exponentially
with `n` and the factors cancel, so the identity residual `max|A A^-1 - I|` of random skew
planes grew from 3e-14 (n = 32) through 4e-12 (c64, n = 64) to 1e-8 (c64) / 1e-9 (f64) at
n = 128 while pfapack stayed at 3e-14 (the c64 n = 128 difference to pfapack, 1.2e-7, exceeded
`16 n eps cond` = 2.3e-9). The cause was neither the pivot rule (the factor and the Pfaffian
agree with pfapack to 1e-14) nor the bound or the condition estimate (`cond ~ 1e3`). Back
substitution has the stability of `dtrtri`: n = 128 now gives residual 1e-14..3e-14 and
difference to pfapack 4e-15..2e-14, in both dtypes; the `16 n eps cond` bound is unchanged and
`native_inverse_is_stable_at_large_n_real_and_complex` also bounds the residual.

Cost: about 100 session operations per elimination step (137 / 568 / 1631 / 2482 operations for
`n = 2 / 6 / 16 / 24`; `MVMC_GPU_NATIVE_OPCOUNT=1` prints the count); 10 to 15 us per operation
when the tensors are tiny (`n = 16`, 8 planes: 17 to 21 ms) and about 1 ns per element-operation
when they are large, so it is 20 to 400x slower than the pfapack loop (rows `-`: not run) and
only a semantic reference. `TENFERRO_PROFILE_EAGER_OP_AGG` instruments only `tenferro-ad` eager
tensors, not session operations (`TENFERRO_PROFILE_CPU_SESSION` exists for whole-session
sections), so the counts are our own. The `ExtensionOp` backend pays about 8 ms per call for the
runtime build and graph compile plus one input and one output copy, then runs pfapack at full
speed: 1.2 to 2.6x of the pfapack loop for 512 to 4096 planes. It is the route that keeps the
fast kernel inside a tenferro session.

Gaps found while writing both (candidates for upstream requests, none filed here):

* no argmax / first-index-of-max operation; worked around with four operations
  (tensor4all/tenferro-rs#1976);
* no per-matrix row/column gather or swap with batch-varying indices (StableHLO `gather` here
  has no operand batching dims); worked around with one-hot masks, `select` and `reduce_sum`, an
  `O(n^2 P)` pass per step instead of `O(n P)` (tensor4all/tenferro-rs#2008);
* `Tensor` is not `Clone` (only a deep `duplicate`), so reusing an intermediate in two places
  is awkward; a cheap shared handle would avoid it;
* `tril`/`triu` have an unspecified axis convention for a trailing batch axis, so static 0/1
  masks plus `select` were used; elementwise operations need explicit `broadcast_in_dim` (no
  implicit broadcasting);
* `GraphCompiler::compile_with_input_specs` takes one output, so a multi-output extension op
  with a runtime-bound input cannot be compiled once and re-run (the plane tensor is attached as
  a concrete leaf: one input copy and a compile per call);
* `|re| + |im|` (`izamax`) needs complex-to-real `cast`; it projects the real part as needed,
  but there is no documented `real`/`imag` operation;
* `define_extension_runtime!` needs a runtime built with `runtime_engine_registration_with_id`
  plus `install_extension_module`; the required wiring is not documented next to the macro;
* CUDA: no pinned or asynchronous host transfers (0.5 to 0.8 GB/s measured), `Module` is
  `!Send` and so cannot be cached in `Session::resource`, and `raw` has no way to opt in to more
  than 48 KB of dynamic shared memory (`cuFuncSetAttribute`).

### 11.6 Validation harness (#424) and the invM convention

`mvmc_gpu::stages::BatchedStages` implements `PfaffianStages` on every
batched backend (section 14; `into_stage_backend()` composes it with the C-order SR stages). CPU variants
(`CpuPfapack`, rayon, `ExtensionOp`, tensor-native) run the teacher-forced replay with the flip
detector in normal CI (`crates/mvmc-gpu/tests/accel_harness.rs`: 0 flips, 0 defects, Pfaffian
bit-identical for pfapack/`ExtensionOp`, inverse max abs 5.7e-14 for tensor-native, minimum
decision margin 5.1e-3); the CUDA kernel runs the same replay in the gate
(`pfaffian_gate.rs`, `cuda_pfaffian_passes_the_validation_harness_replay`).

Sign convention: the harness oracle returns the true inverse `X^-1` (`X * inv = I`), and so does
`pfaffian_inverse_batched` (`utu2inv` output, no flip), so the mapping to the harness is the
identity (`inv_convention_matches_the_oracle` checks it bitwise against `COrderPfaffian`). mVMC's
`invM` is the negative: `calc_m_all_real` / C `CalculateMAll` end with
`M_DSCAL(&nsq, &minus_one, invM, &one)`, hence `invM = -inv` while `pf` is unchanged.
A production integration must apply that flip when storing into the sampler tables.

### 11.7 Reproduce

```sh
cargo nextest run -p mvmc-gpu --cargo-profile test-fast        # CPU, ExtensionOp, tensor-native
MVMC_RS_CUDA_GATE=1 scripts/run_cuda_gate.sh docker            # GPU gate incl. pfaffian_gate
scripts/run_pfaffian_bench.sh docker                           # benchmark, writes the CSV
```


## 12. SR pathfinder through tenferro (issue #421)

Status: implemented; opt-in (`MVMC_RS_SR_BACKEND=tenferro`), C order stays the default and the
parity oracle. This also starts the stage-level backend trait of section 5.2.

### 12.1 What exists

* `crates/mvmc-core/src/sr_backend.rs`: trait `SrStages` (formerly `SrBackend`, section 14) with the stages `gram_real`,
  `gram_complex`, `assemble_s_g`, `cholesky_solve`, `cg_local_product`. `COrderSr` is the former
  inline code moved behind the trait without changing an operation (default outputs are
  byte-identical, checked by the full fixture suite). `TenferroSr` uses tenferro eager ops:
  `dot_general` (Gram, CG, outer product), elementwise `sub`/`mul` (S and g assembly),
  `cholesky` plus two `triangular_solve` (solve). It is generic over the eager runtime and the
  placement (`Placement::Host` / `Placement::Device`); `gpu/mvmc-gpu-cuda` builds it on the CUDA
  runtime (`sr_bench`, gate test `cuda_gate_sr_stages_match_c_order`).
* Hook points: `finalize_oo_store[_real]` (Gram), `sr.rs` (S/g and solve, real and complex
  layouts), `sr_cg.rs` (`SampledSrOperator`: one backend handle and one operand version per
  solve). The MPI reduction, the sequential dots and the mean/diagonal corrections of the CG
  product stay on the host in C order.
* tenferro lessons applied: faer provider only (`cpu-blas` breaks CPU linalg; `autodiff` is
  required for `EagerTensorLinalgExt`); one process-wide runtime built on first use and one
  handle per SR solve; the constant CG operand (`[O_r | O_i]`) is cached under an explicit
  version counter (`TenferroStats::cg_uploads`); host tensors are never created for device
  runs (scalars are uploaded as constant vectors, which CUDA requires); results are explicit
  errors, never a silent C-order fallback.
* Not used: `ConcreteEinsumPlan` + `execute_into`. All contractions are single binary
  `dot_general` calls, whose plan is built inside the eager runtime; a plan adds nothing for
  one-shot shapes and the per-call cost is dominated by tensor upload (below).

### 12.2 Complex Gram sample order (`observables.rs`, formerly `sr_store_gram_julia`)

The authoritative complex finalizer sums every Gram entry sequentially over samples. A general
GEMM may reassociate the sample sum and change both the rounding and the sign of exact zeros,
which changes the direct SR input even for an identical RNG trajectory. Decision: the default
path keeps the sequential order (`c_order_gram_complex`); the tenferro path uses blocked GEMM
order and is validated per entry by `2 gamma_{samples+3} sum |O_is||O_js|`, i.e. only the
reassociation error is allowed. The real Gram already follows C/Julia (`dsyrk`, upper triangle
mirrored); the tenferro real Gram mirrors the upper triangle the same way.

### 12.3 Equivalence and measurements

Tolerances and derivations are in `docs/NUMERICAL_COMPARISONS.md` (row "tenferro SR backend vs
C order"). Measured: Gram, solve and CG product differ from C order by 1e-15..1.5e-13 relative;
S and g assembly agree exactly. CG runs amplify any rounding (one-ulp sample noise moves the
C-order solution by 1e-4..7e-4 at `max_iterations = n`), so step-by-step CG trajectories are not
compared; the test bounds the backend difference by that measured spread.

CPU (release, 1 thread, `cargo run --release -p mvmc-core --example sr_backend_bench`, median of
7, ms, C order / tenferro cpu-faer):

| NPara | samples | Gram | S,g assembly | Cholesky solve | CG matvec |
|---:|---:|---|---|---|---|
| 388 | 300 | 2.18 / 8.72 | 0.69 / 12.02 | 2.14 / 6.01 | 0.113 / 0.238 |
| 1000 | 2000 | 82.95 / 210.66 | 3.82 / 40.68 | 13.89 / 29.27 | 2.300 / 2.353 |
| 3000 | 300 | 116.94 / 408.40 | 139.69 / 512.10 | 288.80 / 566.83 | 0.539 / 0.819 |
| 3000 | 6000 | 1402.24 / 5334.11 | 138.81 / 520.99 | 290.69 / 575.61 | 25.661 / 28.631 |

On one CPU thread the tenferro path is 1.1-4x slower than OpenBLAS/LAPACK: faer GEMM instead of
SYRK (half the flops), and every stage pays eager tensor upload and download (about 1.5 ms per
388x388 tensor, against 0.2 ms for the elementwise op itself). It is a feasibility path, not a CPU
speed-up; the benefit is the device run below. `TENFERRO_PROFILE_EAGER_OP_AGG=1` only reports
`nary_op` sections, so the per-op costs above were measured directly.

CUDA (2x RTX 3060, FP64 at 1/64 of FP32 rate, driver 580.178.04, CUDA 13.0 driver API, NVRTC
12.9, cuBLAS 12.9.2, cuSOLVER 11.7.5, tenferro 0.7.1; `scripts/run_cuda_gate.sh docker`, median of
3 after 1 warm-up, host-to-host ms per stage including upload, synchronized compute and
download; the machine has 36 CPU threads that the C-order and tenferro-CPU columns may use):

| NPara | samples | Gram C / CUDA | Cholesky solve C / CUDA | CG matvec C / CUDA (operand upload once) |
|---:|---:|---|---|---|
| 388 | 300 | 7.41 / 2.99 | 2.81 / 30.94 | 0.10 / 0.98 (0.52) |
| 1000 | 2000 | 128.2 / 82.6 | 28.2 / 14.8 | 3.32 / 0.59 (64.6) |
| 3000 | 300 | 241.0 / 363.3 | 614.9 / 439.9 | 0.66 / 0.78 (11.5) |
| 3000 | 6000 | 3024.6 / 1246.5 | 370.4 / 339.3 | 28.4 / 1.31 (467) |

All CUDA stages agree with C order to <= 6.5e-15 relative (bounds 1e-8..1e-15, all "ok"). The
CUDA gain appears for the large-sample Gram and the cached CG matvec; solve is at parity at
`NPara` 3000, and S/g assembly is transfer-bound (4 uploads of `n^2` doubles), which is why the
next step is device residency of `OO`/`S` across stages rather than faster single stages.

## 13. Device-resident lock-step multi-walker sampler (issue #434)

Status: implemented for the real (`f64`) normal-mode sampler with hopping and exchange updates
(Hubbard chain and Heisenberg chain), validated on the RTX 3060 and benchmarked end to end.
Complex and FSZ modes, the measurement stage and MPI QP splitting are not covered. Related to
#417, #422, #423, #425, #432.

### 13.1 Design

The sampler loop `vmc_make_sample_real` is not rewritten. Its five Pfaffian operations sit behind
the trait `RealPfStage` (`crates/mvmc-core/src/sampling/stage.rs`): the ratios of a proposed
one- or two-electron move, the rank-one and rank-two inverse updates of an accepted move, and the
periodic recomputation (`CalculateMAll`). `CpuStage` runs the existing flat kernels in place (the
production path, same arithmetic, monomorphized, no dispatch cost). Everything else stays in the
unchanged host code, in this order: update-type draw, candidate generation, projection counters,
`log_proj_ratio`, the Metropolis weight and its SFMT draw, configuration bookkeeping. Draw order
and count are therefore the serial sampler's by construction.

`crates/mvmc-core/src/device_sampler.rs` runs `W` walkers in lock-step: one host thread per
walker (each owns its `SamplingWalker`: data, state, SFMT stream seeded `RndSeed + w`, the C group
seed of #425) executes the sampler with a `WalkerStage`, which turns the five operations into
requests to a `DeviceService` owned by the calling thread:

| sampler step | request | host-device traffic |
| --- | --- | --- |
| initial tables | `Begin` (slow lane) | configuration `[n]`, initial Pfaffians `[NQP]`; the Slater table once per distinct wavefunction |
| proposal ratios | `Hop` / `Exchange` (blocking) | move: slot, site (and a second slot for exchange); back: `NQP` ratios |
| accepted move | `Accept` (not blocking) | the walker index (the device remembers the pending move) |
| `CalculateMAll` | `Recompute` (slow lane) | configuration; back: `NQP` Pfaffians |

A **round** fires when every live walker is parked either on a proposal or on a slow-lane
request (pending or in flight): the fast lane applies the round's accepted moves, runs all
proposals and returns the ratios. Slow-lane requests (begin, recompute) run asynchronously on up
to 8 slow lanes so that their latency (a batched Pfaffian/inverse is hundreds of microseconds to
milliseconds) does not stall the other walkers' proposals. Walkers that reject a candidate on
the host (`continue` in the sampler) simply reach their next request later. Results never depend
on which walkers share a round, nor on thread timing: a walker has at most one blocking request
outstanding, and its accept precedes its next request. Replies are delivered through a wake tree
(a futex wake of hundreds of walkers from the single service thread costs more than a device
round), walkers spin for 50 us when they do not outnumber the cores and park otherwise.

The CUDA service (`gpu/mvmc-gpu-cuda/src/device_sampler.rs`, kernels in `sampler_kernels.cu`)
keeps resident per walker: the inverses `[NQP][n*n]` in **mVMC's convention `invM = -X^-1`**, the
Pfaffians, the accepted configuration and the pending move; the Slater table is resident once per
distinct wavefunction (walkers share it when their parameters are bit-equal). What crosses the
link per round is the request ints (one pinned upload) and the `NQP`-vectors of the proposals
(one pinned download); Slater planes are assembled on the device from the configuration.

* `k_propose`: one block per (proposal, QP). The rank-one ratio `-pf * sum_j invM[msa,j] S[rsa,rs_j]`
  and the exchange ratio (`two_ratio_real::<false>`, including the Julia `@turbo` reduction tree with
  explicit `fma`) are the CPU formulas. The products of each serial sum are formed in parallel and
  summed in index order by one thread, so the rounding equals the CPU's.
* `k_accept`: rank-one (`update_one_real`) and rank-two (`update_two_real`) inverse updates in the
  CPU operation order, one block per (accepted walker, QP), then `k_commit_moves`.
* `k_assemble` + `pfinv_f64` (the #423 kernel) + `k_commit_recompute`: the planes of
  `assemble_inv_m_real` (`-S[rs_i][rs_j]`), the batched Pfaffian/inverse, and the commit with the
  sign flip of `calc_m_all_real` (`invM = -inv`). On a failed plane the resident tables are left
  unchanged and the host sees the failure like `calc_m_all_real(..).is_err()`.
* NVRTC runs with `--fmad=false` (explicit `fma()` reproduces Rust's `mul_add`).

**Sign convention.** `pfaffian_inverse_batched` (#423) returns the true inverse `X^-1`, the
convention of the #424 harness oracle. mVMC's `invM` is its negative (`M_DSCAL(-1)` at the end of
`CalculateMAll`/`calc_m_all_real`). The device stores `invM` so that the ratio and update kernels
are the CPU formulas verbatim; the flip is applied in `k_commit_recompute`.

**Transfers.** The host-device copies sit behind `TransferPath` (`PinnedTransfer`: the pool and
asynchronous copies of #432; `PageableTransfer`: the cudarc baseline). Compute and copies go
through cudarc on the service's own streams, bypassing tenferro's `upload`/`download` as the
#432 measurements recommend; the resident buffers are therefore owned by one allocator. The gate
test `pinned_helper_raw_copies_work_on_tenferro_raw_session_addresses` validates the `*_raw`
helpers of #432 against device addresses owned by a tenferro raw session (a `DeviceBytes`
workspace uploaded and downloaded through `upload_raw`/`download_raw` while tenferro launches
the kernel). The sampler keeps its own seam `RealPfStage` (it needs the finer per-step operations, which
the batched `PfaffianStages::pfaffian_inverse_batch` of the unified `StageBackend` cannot
express); the only other backend interface it touches is `PfaffianStages` through
`mvmc_gpu::stages::BatchedStages` (section 14).

**Wavefunction sharing.** The benchmark and the gate build all walkers on the wavefunction
(parameters) of walker 0 with independent chains, the physically relevant multi-walker case.
Walkers with different parameters (the `init_parameter` random initialization of the no-file path
differs per seed) get one resident Slater table each; the service supports both.

**Stale host tables (issue #454).** The device holds the current inverses, so after a device run
the host `inv_m_real` is out of date. This is tracked explicitly: `run_lockstep_real` calls
`InvMColMajor::mark_stale` on each walker's `inv_m_real` when the run ends. Every read of a stale
plane (`as_slice`, `get`, `qp_matrix_slice`, `pad_slot`, plane views) and every incremental update
(`as_mut_slice`, `set`) panics with "stale inverse table read" naming the reason; the
full-overwrite writer `qp_matrix_slice_mut`, which every `calc_m_all_*` recompute uses, validates
the plane again, so the next `vmc_make_sample_real` call (which recomputes all planes first)
restores a valid table. A table never marked stale carries no tracking cost beyond an `Option`
check and the default CPU path is unchanged. The host Pfaffian buffer `pf_m_real` needs no flag:
the stage keeps it current (hop and recompute copies as before, and since #454 the exchange accept
copies the proposed Pfaffians too, matching the CPU in-place update). Audit of the readers: the
sampler, measurement, Lanczos, SR derivative and output stages all run after a `calc_m_all_*`
recompute on the CPU path; none is reached from `run_lockstep_real` today, and a future reader
that skipped the recompute now fails loudly. A caller that needs the device tables downloads them
(`download_walker` in the test API).

### 13.2 Validation

Normal CI (`crates/mvmc-core/tests/device_sampler.rs`, no GPU): the `HostService` (the same flat
CPU kernels behind the same request protocol, per-walker resident tables) driven through the
lock-step runner equals the plain CPU sampler for `W = 1` and `W = 3`, Hubbard hopping (L = 16) and
Heisenberg exchange: sampler statistics, every recorded Metropolis `(weight, draw)` pair, the
final SFMT state (all 624 words and the position), the consumed word count, the working and saved
configurations, bit for bit. A teacher-forced run reports 0 flips, 0 draw mismatches and a
maximum weight difference of exactly 0; a negative test forces one reference decision to
disagree and the detector reports the flip and classifies it as a defect.

GPU gate (`gpu/mvmc-gpu-cuda/tests/sampler_gate.rs`, `MVMC_RS_CUDA_GATE=1
scripts/run_cuda_gate.sh docker`, RTX 3060): per case the CPU sampler runs once and records its
decisions; then (a) a teacher-forced run on the device with both transfer paths follows the CPU
decisions and reports flips, defects (a flip whose margin `|w_cpu - u|` exceeds the weight
difference or the `1e-12 + 1e-10 w` bound), draw mismatches, the weight difference and the
smallest margin; the walker's final RNG state and configuration must equal the CPU's (they do:
the draws are host draws); (b) a free run decides on the device weights and must reproduce the
CPU decision sequence, final configuration and RNG state; (c) the resident inverses and Pfaffians
after the run are compared with the CPU tables.

| case | decisions | flips | defects | max rel. weight diff | min margin | resident invM rel. diff | resident pf rel. diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hubbard L=16, W=4, hopping | 1920 | 0 | 0 | 5.9e-12 | 1.9e-4 | 2.7e-14 | 5.7e-14 |
| Hubbard L=32, W=3, hopping | 2880 | 0 | 0 | 5.9e-9 | 3.4e-4 | 1.5e-13 | 1.2e-13 |
| Heisenberg N=6, W=4, exchange | 1200 | 0 | 0 | 3.6e-13 | 6.0e-4 | 7.4e-17 | 0 |

The ratio and update kernels are bit-compatible with the CPU formulas; the weight differences
come from the recomputation: the device inverse of `pfinv_f64` rounds differently from the
production PfaPack path (BLAS `trmm`, panel product), and the difference is carried through the
chain of rank-one updates between recomputations (`2.7e-14` to `1.5e-13` in the final inverse; the
L=32 weight difference of `5.9e-9` is a relative difference of one tail weight, absolute `9e-10`).
It is three to five orders of magnitude below the smallest decision margins (`1.9e-4` here, `1.5e-7`
on the Heisenberg-chain PhysCal fixture of #425), so a flip needs a decision within `1e-9` of its
draw; the detector reports none and would report any as a defect. The `*_raw` interoperability
test passes.

### 13.3 Benchmark

Environment: Intel Xeon E5-2699 v3 (36 hardware threads) shared with other jobs (load average
11 to 36 at the start and end of the runs, see the CSV headers; the numbers below carry that noise,
two complete campaigns agreed within about +-30 %, the L=64 `W=512` row of the 4-core table flipped
between 0.95x and 2.1x), 2x NVIDIA GeForce RTX 3060 (sm_86, 12 GB, device 0), driver 580.178.04,
CUDA driver API 13.0, container CUDA toolkit 12.9.2 (NVRTC 12.9), rustc 1.98.0, tenferro 0.7.1,
Linux x86_64. Docker image `tenferro-benchmark-cuda:full-verify-20260822`.

What is timed: one *call* is `vmc_make_sample_real` of every walker, the C `VMCMakeSample` of one
sample series: `(NVMCWarmUp + NVMCSample) * Nsite` hop attempts per walker (about 3000 per walker
and call; `NVMCSample` is chosen per size), the initial table construction (the CPU builds
its tables for the initial configuration in both variants; the device additionally builds its
resident tables from the configuration: a device `Begin` batch), and the periodic
recomputations. Hubbard chain, half filling, `Lsub = 4`, `U = 4`, `NSPGaussLeg = 8` (`NQP = 8`),
real normal mode, hopping updates, inputs `benchmark/hubbard_chain/inputs/hubbard_chain_L{16,32,64,128}`
(`L128` is new, generated with the Rust StdFace port like `L64`). All `W` walkers share one
wavefunction and have independent SFMT streams (`RndSeed + w`). Each method builds its own walkers,
runs one untimed warm-up call (burn-in), then three timed calls (median). Measurement
(`VMCMainCal`) is not part of the call. Thread creation per call (one thread per walker) and, for the
device, service construction excluded, Slater-table upload included.

* **CPU 1 thread**: the walkers one after the other on one thread (only for `W <= 64`).
* **CPU multichain**: one thread per walker with up to all host cores, the #425 execution model
  (independent walkers, single-threaded inside; BLAS pinned to one thread).
* **CUDA device-resident**: this design, pinned asynchronous transfers (#432), one host thread per
  walker, the Pfaffian stages on the GPU.

The 4-core tables run the same program in a container restricted to cores 0-3 (`docker
--cpuset-cpus=0-3`): a GPU attached to a small host, the CPU baselines and all walker threads of the
device run share exactly those cores. Columns "CUDA / CPU" are wall-time ratios (above 1 the device
is slower). The CSVs are `benchmark/gpu_device_sampler/results/device_sampler_cores{all,4}.csv`
(plus `_pageable.csv`); the metadata block (device report, host cores, load average) is in the
CSV/log header and in `results/device_sampler.md`.


**Wall time of one sampling call, 36 host cores (all of them; shared host)**

| L | W | CPU 1 thread | CPU multichain | CUDA device-resident | hops/s CPU multichain | hops/s CUDA | CUDA / CPU-multichain time | CUDA / CPU-1-thread time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 0.010 s | 0.010 s | 0.090 s | 298k | 33k | 8.96x | 9.24x |
| 16 | 8 | 0.065 s | 0.014 s | 0.142 s | 1.69M | 169k | 10.04x | 2.17x |
| 16 | 64 | 0.503 s | 0.082 s | 0.932 s | 2.34M | 206k | 11.40x | 1.85x |
| 16 | 512 | - | 0.644 s | 1.983 s | 2.38M | 773k | 3.08x | - |
| 32 | 1 | 0.020 s | 0.020 s | 0.106 s | 151k | 28k | 5.40x | 5.24x |
| 32 | 8 | 0.148 s | 0.024 s | 0.182 s | 1.01M | 131k | 7.72x | 1.23x |
| 32 | 64 | 1.079 s | 0.077 s | 0.852 s | 2.46M | 224k | 11.00x | 0.79x |
| 32 | 512 | - | 0.577 s | 2.142 s | 2.64M | 711k | 3.71x | - |
| 64 | 1 | 0.050 s | 0.052 s | 0.119 s | 56k | 25k | 2.28x | 2.36x |
| 64 | 8 | 0.438 s | 0.058 s | 0.206 s | 404k | 114k | 3.54x | 0.47x |
| 64 | 64 | 4.564 s | 0.249 s | 0.914 s | 757k | 206k | 3.67x | 0.20x |
| 64 | 512 | - | 1.725 s | 4.070 s | 874k | 370k | 2.36x | - |
| 128 | 1 | 0.127 s | 0.117 s | 0.135 s | 25k | 22k | 1.15x | 1.07x |
| 128 | 8 | 1.006 s | 0.160 s | 0.301 s | 147k | 78k | 1.88x | 0.30x |
| 128 | 64 | 8.092 s | 0.756 s | 2.750 s | 249k | 69k | 3.64x | 0.34x |
| 128 | 512 | - | 4.941 s | 10.119 s | 305k | 149k | 2.05x | - |

**Anatomy of the device run, 36 host cores (all of them; shared host)**

| L | W | wall | passes | ms/pass | service thread: wait for walkers / round / reply | in round: stage / upload / launch / device wait | slow batches |
|---|---:|---:|---:|---:|---|---|---:|
| 16 | 1 | 0.090 s | 2911 | 0.031 | 14.6 / 66.0 / 0.9 ms | 0.9 / 15.2 / 27.9 / 20.7 ms | 63 |
| 16 | 8 | 0.142 s | 3031 | 0.047 | 11.6 / 109.7 / 12.0 ms | 2.8 / 20.4 / 49.5 / 35.1 ms | 479 |
| 16 | 64 | 0.932 s | 2917 | 0.319 | 371.5 / 391.3 / 101.7 ms | 24.0 / 91.1 / 157.4 / 108.3 ms | 2210 |
| 16 | 512 | 1.983 s | 2921 | 0.679 | 866.4 / 487.5 / 289.1 ms | 59.6 / 91.1 / 130.2 / 188.9 ms | 2892 |
| 32 | 1 | 0.106 s | 2720 | 0.039 | 17.9 / 76.4 / 1.1 ms | 0.9 / 15.4 / 28.2 / 30.8 ms | 32 |
| 32 | 8 | 0.182 s | 2860 | 0.064 | 19.5 / 134.2 / 21.3 ms | 2.7 / 19.9 / 45.7 / 64.1 ms | 225 |
| 32 | 64 | 0.852 s | 2749 | 0.310 | 215.8 / 355.3 / 77.7 ms | 17.5 / 60.2 / 106.8 / 163.9 ms | 1343 |
| 32 | 512 | 2.142 s | 2723 | 0.787 | 640.9 / 780.9 / 235.1 ms | 50.1 / 80.9 / 118.8 / 514.7 ms | 2654 |
| 64 | 1 | 0.119 s | 2382 | 0.050 | 21.3 / 84.5 / 2.6 ms | 0.9 / 15.1 / 27.8 / 39.4 ms | 14 |
| 64 | 8 | 0.206 s | 2538 | 0.081 | 25.2 / 145.0 / 27.1 ms | 2.5 / 17.2 / 38.8 / 84.9 ms | 100 |
| 64 | 64 | 0.914 s | 2426 | 0.377 | 183.1 / 507.7 / 68.9 ms | 14.4 / 48.3 / 85.0 / 354.4 ms | 677 |
| 64 | 512 | 4.070 s | 2400 | 1.696 | 766.6 / 2313.8 / 264.1 ms | 50.1 / 73.7 / 114.2 / 2061.2 ms | 2074 |
| 128 | 1 | 0.135 s | 1797 | 0.075 | 26.2 / 82.4 / 1.8 ms | 0.7 / 10.8 / 19.1 / 51.0 ms | 5 |
| 128 | 8 | 0.301 s | 2015 | 0.150 | 42.9 / 207.2 / 23.9 ms | 2.0 / 14.6 / 32.8 / 156.5 ms | 36 |
| 128 | 64 | 2.750 s | 1824 | 1.508 | 1212.5 / 1025.9 / 114.0 ms | 17.6 / 69.8 / 110.7 / 820.6 ms | 262 |
| 128 | 512 | 10.119 s | 1810 | 5.591 | 1571.0 / 6430.9 / 200.2 ms | 47.2 / 69.6 / 103.2 / 6199.8 ms | 934 |

**Transfer path (pinned async vs pageable), 36 host cores (all of them; shared host)**

| L | W | pinned async | pageable | pageable / pinned |
|---|---:|---:|---:|---:|
| 16 | 8 | 0.142 s | 0.466 s | 3.29x |
| 16 | 64 | 0.932 s | 0.981 s | 1.05x |
| 64 | 8 | 0.206 s | 0.319 s | 1.55x |
| 64 | 64 | 0.914 s | 1.500 s | 1.64x |

**Wall time of one sampling call, 4 host cores (docker --cpuset-cpus=0-3)**

| L | W | CPU 1 thread | CPU multichain | CUDA device-resident | hops/s CPU multichain | hops/s CUDA | CUDA / CPU-multichain time | CUDA / CPU-1-thread time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 0.009 s | 0.009 s | 0.099 s | 317k | 30k | 10.49x | 10.60x |
| 16 | 8 | 0.072 s | 0.034 s | 0.304 s | 704k | 79k | 8.93x | 4.22x |
| 16 | 64 | 0.573 s | 0.200 s | 1.079 s | 957k | 177k | 5.39x | 1.88x |
| 16 | 512 | - | 1.360 s | 5.553 s | 1.13M | 276k | 4.08x | - |
| 32 | 1 | 0.021 s | 0.020 s | 0.164 s | 146k | 18k | 8.02x | 7.99x |
| 32 | 8 | 0.161 s | 0.073 s | 0.441 s | 324k | 54k | 6.00x | 2.74x |
| 32 | 64 | 1.264 s | 0.324 s | 1.056 s | 588k | 180k | 3.26x | 0.84x |
| 32 | 512 | - | 2.590 s | 5.060 s | 588k | 301k | 1.95x | - |
| 64 | 1 | 0.051 s | 0.051 s | 0.123 s | 58k | 24k | 2.42x | 2.45x |
| 64 | 8 | 0.396 s | 0.111 s | 0.286 s | 213k | 82k | 2.59x | 0.72x |
| 64 | 64 | 3.233 s | 0.873 s | 1.243 s | 216k | 152k | 1.42x | 0.38x |
| 64 | 512 | - | 6.879 s | 14.496 s | 219k | 104k | 2.11x | - |
| 128 | 1 | 0.207 s | 0.207 s | 0.261 s | 14k | 11k | 1.26x | 1.26x |
| 128 | 8 | 1.897 s | 0.729 s | 0.640 s | 32k | 37k | 0.88x | 0.34x |
| 128 | 64 | 14.905 s | 2.650 s | 2.057 s | 71k | 92k | 0.78x | 0.14x |
| 128 | 512 | - | 18.188 s | 13.194 s | 83k | 114k | 0.73x | - |

**Anatomy of the device run, 4 host cores (docker --cpuset-cpus=0-3)**

| L | W | wall | passes | ms/pass | service thread: wait for walkers / round / reply | in round: stage / upload / launch / device wait | slow batches |
|---|---:|---:|---:|---:|---|---|---:|
| 16 | 1 | 0.099 s | 2911 | 0.034 | 17.3 / 73.3 / 1.2 ms | 1.1 / 18.1 / 34.5 / 17.9 ms | 63 |
| 16 | 8 | 0.304 s | 2961 | 0.103 | 46.6 / 173.5 / 80.9 ms | 6.3 / 45.5 / 100.9 / 16.3 ms | 475 |
| 16 | 64 | 1.079 s | 2917 | 0.370 | 461.8 / 320.6 / 173.4 ms | 16.7 / 59.5 / 111.7 / 125.3 ms | 2210 |
| 16 | 512 | 5.553 s | 2921 | 1.901 | 5339.7 / 708.5 / 937.5 ms | 80.8 / 150.5 / 217.1 / 234.7 ms | 2892 |
| 32 | 1 | 0.164 s | 2720 | 0.060 | 33.8 / 111.3 / 4.5 ms | 2.2 / 31.9 / 60.3 / 13.9 ms | 32 |
| 32 | 8 | 0.441 s | 2778 | 0.159 | 116.5 / 184.0 / 120.7 ms | 7.1 / 49.9 / 101.9 / 20.4 ms | 224 |
| 32 | 64 | 1.056 s | 2729 | 0.387 | 372.1 / 332.8 / 150.8 ms | 15.3 / 50.2 / 90.0 / 171.3 ms | 1358 |
| 32 | 512 | 5.060 s | 2722 | 1.859 | 2747.7 / 831.0 / 799.9 ms | 54.6 / 86.4 / 126.4 / 543.3 ms | 2655 |
| 64 | 1 | 0.123 s | 2382 | 0.052 | 20.5 / 83.6 / 2.6 ms | 1.1 / 15.8 / 29.5 / 35.7 ms | 14 |
| 64 | 8 | 0.286 s | 2490 | 0.115 | 51.5 / 154.3 / 58.7 ms | 3.6 / 25.5 / 52.5 / 70.3 ms | 99 |
| 64 | 64 | 1.243 s | 2408 | 0.516 | 343.8 / 405.3 / 118.7 ms | 12.2 / 36.1 / 67.0 / 285.2 ms | 688 |
| 64 | 512 | 14.496 s | 2384 | 6.080 | 9038.5 / 2130.9 / 2315.8 ms | 88.4 / 235.1 / 281.9 / 1498.5 ms | 2078 |
| 128 | 1 | 0.261 s | 1797 | 0.145 | 113.9 / 112.6 / 12.7 ms | 1.7 / 26.2 / 43.7 / 38.8 ms | 5 |
| 128 | 8 | 0.640 s | 1879 | 0.340 | 209.7 / 238.8 / 120.6 ms | 5.1 / 39.3 / 70.9 / 120.2 ms | 36 |
| 128 | 64 | 2.057 s | 1839 | 1.119 | 499.4 / 982.7 / 60.6 ms | 8.9 / 26.0 / 46.1 / 898.6 ms | 265 |
| 128 | 512 | 13.194 s | 1807 | 7.302 | 2980.5 / 5471.6 / 665.8 ms | 34.0 / 48.7 / 74.6 / 5305.6 ms | 934 |

**Transfer path (pinned async vs pageable), 4 host cores (docker --cpuset-cpus=0-3)**

| L | W | pinned async | pageable | pageable / pinned |
|---|---:|---:|---:|---:|
| 16 | 8 | 0.304 s | 0.268 s | 0.88x |
| 16 | 64 | 1.079 s | 0.913 s | 0.85x |
| 64 | 8 | 0.286 s | 0.359 s | 1.26x |
| 64 | 64 | 1.243 s | 1.617 s | 1.30x |


### 13.4 What the numbers say

* **The device-resident sampler does not beat the host on this machine.** With all 36 cores the
  CPU multichain runner is 1.15x (L=128, `W=1`) to 11.4x (L=16, `W=64`) faster, and still 2.1x to 3.7x
  faster at `W=512`. The device is faster than **one** core for `W >= 8` at `L >= 64` (L=64, `W=64`: 5x;
  L=128: about 3x) and from `W = 64` at L=32, which is the replacement of a handful of cores, not of 36.
* **Only a small host changes the picture, and only at n = 128.** With 4 host cores the device wins
  at L=128 for every `W >= 8`: 1.14x (`W=8`), 1.28x (`W=64`), 1.37x (`W=512`) faster than the
  4-core CPU runner; at n <= 64 it still loses (1.4x to 10x), with the L=64 `W=512` point at parity
  in one campaign and 2.1x slower in the other. This is the Amdahl bound of section 3.2: on the Hubbard-chain
  profile the Pfaffian share of the sampler (`UpdateMAll`, `CalculateNewPfM2`, recomputation) is
  about 32 % at L=16 and 45 % at L=64, so removing it entirely (infinitely fast device, free
  transfers) cannot give more than about 1.5x to 1.8x; the rest (candidate generation, projection
  counters and `log_proj_ratio`, SFMT, the acceptance weight) stays on the host by the RNG-parity
  contract and costs 3.3 us (L=16), 6.7 us (L=32), 17 us (L=64) and 43 us (L=128) per attempted hop
  per core (the `CPU 1 thread`, `W=1` rows).
* **Fixed cost per round.** At `W=1` a round costs 31 us (L=16) to 70 us (L=128): about 1 us of
  staging, 5 to 7 us of upload enqueue, 10 to 15 us of kernel launches (three to five launches
  through cudarc), 6 to 25 us of device execution and download. A CPU hop costs 3 to 43 us, so
  single walkers can never win (the `W=1` column is 9x slower at L=16 and 1.1x at L=128, where a
  hop is expensive enough to hide the round).
* **Where the time goes at scale (L=128, `W=512`).** The fast lane's device wait is 61 % of the wall
  time with 36 cores and 40 % with 4 cores. The `nsys` profile of L=128, `W=64` attributes 64 % of the GPU time to
  `pfinv_f64` (5.2 ms per batch of 8 planes: the serial LTL^T steps and column loops of a single
  block per plane, `B=1` of the #423 benchmark), 28 % to `k_accept` (0.32 ms per launch) and 5 % to
  `k_propose` (53 us). The slow lanes overlap with the fast lane but compete with it for the SMs.
  The service thread then waits 20 % of the wall time for the walkers' host work and 5 % for
  reply delivery. The kernels are straightforward ports, not tuned: a shared-memory LTL^T for
  `n <= 64`, a parallel triangular inverse and a tiled rank-two update are the obvious next steps;
  none changes the Amdahl bound above.
* **Host coordination matters.** One thread per walker needs a futex wake per walker per round
  (hundreds of microseconds at 64 to 512 walkers, hence the wake tree) and degenerates when
  walkers greatly outnumber the cores (512 walkers on 4 cores took 107 s with timed parking before
  the plain `park`, 4 to 14 s after). A resumable state machine per walker, driven by a few worker
  threads, would remove that cost; it duplicates the driver's loop and was not attempted here.
* **Transfers.** Pinned asynchronous copies make the end-to-end run 1.26x to 1.64x faster than
  pageable copies at L=64 (`W = 8, 64`, both host configurations) and are equal within noise at
  L=16 (0.85x to 1.05x; one 3.3x outlier of the 36-core campaign was a noisy run): the effect
  grows with the bytes per round (the recompute configurations and the proposals' `NQP` ratios are
  tens of kilobytes).
* **What would make the device pay.** The remaining host cost per hop, mostly the projection
  counters and `log_proj_ratio` (Gutzwiller/Jastrow), could move to the device without touching the
  RNG contract (the SFMT draws stay on the host; only a scalar log ratio would come back), which is
  the larger lever on hosts with many cores; together with the persistent-kernel or CUDA-graph
  launch path (the 30 to 70 us round floor) and the tuned kernels it is the follow-up to #434. Complex
  and FSZ modes use the same protocol with `Complex64` kernels.


### 13.5 Reproduce

```sh
cargo nextest run -p mvmc-core --cargo-profile test-fast -E 'binary(device_sampler)'   # CPU, host service
MVMC_RS_CUDA_GATE=1 scripts/run_cuda_gate.sh docker                                    # GPU gate (sampler_gate)
scripts/run_device_sampler_bench.sh docker                                             # all host cores
MVMC_RS_SAMPLER_CORES=0-3 MVMC_RS_SAMPLER_BENCH_OUT=$PWD/benchmark/gpu_device_sampler/results/device_sampler_cores4.csv \
  scripts/run_device_sampler_bench.sh docker                                           # 4 host cores
```

## 14. Unified stage backend (issue #437)

Status: implemented. The two overlapping traits of #421 (`SrBackend`) and #424
(`AcceleratedStages`) are replaced by one layering, so the validation harness drives the same
objects that production selects ("validated means deployed").

### 14.1 Layering

* Stage traits, each with host-slice inputs and outputs and `StageError::Unsupported` for a
  stage a backend cannot run (never a CPU fallback):
  * `mvmc_core::sr_backend::SrStages`: Gram, S/g assembly, Cholesky solve, CG product and the
    composite `sr_s_g` stage of the harness;
  * `mvmc_core::stage_backend::PfaffianStages`: batched Pfaffian and inverse
    (`pfaffian_inverse_batch` over `planes` planes with a per-plane `PlaneOutcome`, plus a
    single-plane convenience).
* `mvmc_core::stage_backend::StageBackend<'a>`: the single backend object composing one
  implementation per stage trait (`sr()`, `pfaffian()`, `label()`, `provider()`). Mixed
  backends are expressed by composition (`with_pfaffian`), for example tenferro SR plus the
  CUDA Pfaffian kernel.
* Selection, `StageBackendKind`: C-order CPU (default, parity oracle: `COrderSr` +
  `COrderPfaffian`), tenferro CPU (`TenferroSr` + unsupported Pfaffian) and CUDA
  (`gpu/mvmc-gpu-cuda`: `TenferroSr` on the device + the batched CUDA kernel). Production reads
  `MVMC_RS_SR_BACKEND=c-order|tenferro|cuda[:N]`; `acquire()` returns a handle to a
  process-wide shared instance (non-C-order kinds are built once). CUDA registers through
  `CudaProvider::open_stage_backend`; without the `gpu-cuda` feature or a provider it is an
  error.
* Harness: `accel_validation::{replay, repeatability, bench_stages}` take a `StageBackend`; the
  oracle is `StageBackend::c_order()`. The CUDA gate opens its backend through
  `open_stage_backend(StageBackendKind::Cuda(0))`, the same call production uses.

The CUDA Pfaffian slot uses `PersistentCudaEngine` (`gpu/mvmc-gpu-cuda/src/pfaffian.rs`): a worker
thread keeps one CUDA session and the NVRTC-compiled module alive (`raw::Module` is `!Send`),
because the harness and production issue many small calls and the per-call `CudaEngine` pays
context creation plus NVRTC each time (the first unified gate run took 1209 s instead of 35 s
before this fix).

### 14.2 Where things went

| before | after |
| --- | --- |
| `SrBackend` (#421) | `SrStages` (+ `provider`, `stats`, `sr_s_g`) |
| `SrBackendKind`, `acquire`, `set_sr_backend_override` | `StageBackendKind`, `stage_backend::acquire`, `set_stage_backend_override` |
| `AcceleratedStages::{pfaffian_inverse, sr_s_g}` | `PfaffianStages::pfaffian_inverse[_batch]`, `SrStages::sr_s_g` |
| `CpuOracle` | `COrderPfaffian` + `COrderSr` (`StageBackend::c_order()`) |
| `TenferroCpuStages`, `gpu::stages::EagerStages` | `TenferroSr` (host or device placement) |
| `BatchedStages` (`impl AcceleratedStages`) | `BatchedStages` (`impl PfaffianStages`, `into_stage_backend`) |

### 14.3 Adding a stage (batched local energy #426, device-resident sampler #434)

Define a stage trait next to its data types (`Unsupported` default), add one slot and one
accessor to `StageBackend`, implement it for the C-order object first (the oracle), and extend
the harness replay to compare it. The sampler (#434) should consume `StageBackend` through
`PfaffianStages::pfaffian_inverse_batch` and a new sampler-stage trait rather than any private
trait; the production SR path is the model (`stage_backend::acquire`).

Not done here: the measurement and sampler Pfaffian still call `calc_m_all_*` directly (the
`PfaffianStages` slot is exercised by the harness and gates); wiring it into production is the
#422/#434 work. SR stages are routed in production since #421.

### 14.4 Measurement stage A through the stage backend (issue #422)

`measurement_batch::stage_a_pfaffian` routes the table construction of the batched measurement
(stage A of #428) through `PfaffianStages::pfaffian_inverse_batch`: the real, non-FSZ stage A
assembles the planes of a whole batch in the batched layout `[n, n, NQP * B]` (the formula of
`assemble_inv_m_real`, `X[msj, msi] = -S[rsi, rsj]`), makes one backend call, and stores mVMC's
`invM = -X^-1` with `pf` unchanged (the stage returns the true inverse `X^-1`; the sign flip is
`calc_m_all_child_real`'s `M_DSCAL(.., -1, ..)`). A failing plane (zero pivot, non-finite
Pfaffian, all-zero plane, out-of-range site) fails its whole sample like `calc_m_all_real`.

* Selection: `MVMC_RS_MEASURE_PF_BACKEND=calc-m-all|c-order|tenferro|cuda[:N]` (default
  `calc-m-all`, the unchanged C-order kernels). It is separate from `MVMC_RS_SR_BACKEND`
  because tenferro 0.7.1 has SR stages but no Pfaffian: a backend without the stage, or a
  complex/FSZ mode, is a hard error, never a fallback.
* Validation: `c-order` through the stage is the same PfaPack sequence and is byte-identical to
  the default for batch sizes 1, 3 and all samples (optimization with DH/Gutzwiller/Jastrow and
  PhysCal with Green and Lanczos; `measurement_batch_422.rs`); the table values equal
  `calc_m_all_real` (unit test). The CUDA kernel is validated by
  `measurement_gate.rs` (PhysCal on two Hubbard fixtures, 964 numeric tokens each, worst relative
  difference 1.8e-12 against bound `abs 1e-11 / rel 1e-9`; the sampler and RNG do not depend on
  the measurement backend).
* Not done: complex/FSZ Pfaffian stages, batched local energy/Green/Slater derivative (stage B
  stays per-sample), performance tuning of the batched call (correctness first).

## 15. Device-resident SR pipeline for large NPara (issue #447)

Status: implemented for real parameters (direct Cholesky and CG), validated and benchmarked on the
RTX 3060. Related to #417, #421, #432, #437. Complex parameters need the same stages with
`Complex64` (a second sample matrix in the CG product already works; the complex Gram and
Hermitian Cholesky are not implemented).

### 15.1 Design

`gpu/mvmc-gpu-cuda/src/sr_device.rs` (`DeviceSr`) keeps the SR step on the device between stages.
Only per-step inputs and small outputs cross PCIe:

| stage | where it runs | crosses PCIe |
| --- | --- | --- |
| sample matrix `O` (`[n, samples]`, or the active-component matrix of CG) | resident | once per step, `8 n samples` bytes, pinned write-combined staging in 32 MiB chunks, two buffers so that the host copy overlaps the transfer (#432) |
| Gram `G = O O^T` | cuBLAS `dsyrk` (upper) | nothing |
| S and g assembly with the active-component map, diagonal shift `1 + DSROptStaDel` | kernels `k_assemble_s`/`k_assemble_g`, bitwise the formulas of `COrderSr::assemble_s_g` (`--fmad=false`), reading the upper triangle of `G` | `HO` (`n`) and the map in, nothing out |
| solve | cuSOLVER `dpotrf('U')` + `dpotrs('U')`, the `DPOSV` of C `stcopt_dposv.c`; `info != 0` or a nonfinite solution is `SrDeviceError::SolveFailed` (the host's `Err(())`) | `x` (`NPara` doubles) and two ints out |
| CG | the loop of `SampledSrOperator::solve` ported statement for statement: threshold `tol^2 n^2` left to right, 20-iteration explicit residual refresh, the `beta * delta` recurrence, breakdown exit; the product is two cuBLAS `dgemv` per sample matrix plus a combine kernel (`z = w*y - (mean.x) mean + shift*diag*x`); `daxpy`/`dscal` for the updates | gradient, mean, diagonal in; solution out; three scalars per iteration (`ddot`, host pointer mode) are the only syncs |

There is no CPU fallback: every failure is a typed `SrDeviceError`. S and G never leave the
device in a production step (`download_gram`/`download_s_g` exist for the gate). Device memory
is `8 (n^2 + n_active^2 + n samples)` bytes (direct) or `8 n samples` (CG); a 10^4 x 10^4 step
needs 2.4 GB of the 12 GB.

**Opt-in and the unified backend (#437).** The fused step is two new composite stages of
`SrStages` (`mvmc_core::sr_backend`): `direct_step(&DirectStepInput) -> x` (Gram, S/g assembly,
shift, Cholesky) and `cg_step(&CgStepInput) -> solution` (the CG loop). The C-order backend
implements them as the production composition (`direct_step`'s default composes the backend's own
`gram_real`/`assemble_s_g`/`cholesky_solve`; `cg_step` is `SampledSrOperator::solve_with_stages`,
the production loop, now parameterized over the stage object), so `StageBackend::c_order()` is
the oracle of the device step through the same trait. `ResidentCudaSr` (`gpu/mvmc-gpu-cuda/src/
stages.rs`) delegates every per-stage method to the tenferro CUDA stages and overrides the two
composite stages with `DeviceSr`; `cuda_resident_stage_backend(ordinal)` builds the opt-in
`StageBackend`. The default `MVMC_RS_SR_BACKEND=cuda` backend, the C-order path and the production
call sites are unchanged: production `sr.rs`/`sr_cg.rs` do not yet route to the composite stages
(their S/g assembly consumes the host `OO` array produced earlier by `finalize_oo_store_real`, which
the resident path deliberately never materializes on the host), so selecting the resident backend
in a run is the follow-up; per the priority decision (correct numerics first) this change stops
at the validated, opt-in stages.

### 15.2 Validation against C order (`tests/sr_device_gate.rs`, `MVMC_RS_CUDA_GATE=1`)

Bounds are derived, not tuned:

* **Gram** (step-1 operand): per entry `|G_dev - G_host| <= 2 k eps (|O||O|^T)_ij` with `k` the
  sample count (two different summation orders of `k` products, `gamma_k` each). Observed worst
  ratio observed/bound: `9.8e-3` (n=17), `4.8e-2` (n=129), `6.2e-2` (n=400), `2.4e-2` (n=600).
* **S and g assembly**: bit-identical to the host formula applied to the device's own Gram.
* **Direct solution**: relative error `|dx|/|x| <= 4 n eps kappa(S)` with `kappa` estimated on the
  host (power and inverse iteration through the C-order Cholesky), and the device residual
  `|S x - g|/|g| <= 1e-9`. Observed: `7e-15 / 4.5e-12` (n=17, kappa 3e2), `4.7e-14 / 7.7e-11`
  (n=129), `2.4e-13 / 4.0e-10` (n=400), `2.6e-14 / 2.3e-10` (n=600); residuals `5e-15..9e-14`.
* **Failure semantics**: an indefinite `S` (negative shift) fails on the host (`Err(())`) and on the
  device (`SolveFailed { info > 0 }`); the pipeline stays usable afterwards.
* **CG operator product** (step-1 operand of the CG path): per entry
  `|z_dev - z_host| <= 4 (k + m) eps scale_i`, `scale_i` the sum of the absolute terms of the
  product; observed worst ratio `5.8e-4`, `2.5e-5`, `5.3e-6` (24 to 1125 components, 50 to 1500
  samples).
* **CG solutions and #358.** On a well-conditioned problem (shift 0.5, 3000 samples) the device
  and host iteration counts are equal (29) and the solutions differ by `2.9e-15` (relative). On an
  ill-conditioned problem (120 components, 130 samples, shift `1e-4`, tolerance 0, the regime of
  the #358 analysis where the squared residual cancels to 1e-16) the solutions differ by
  `1.8e-16, 3.4e-16, 7.3e-16, 2.0e-15` after 1, 2, 4, 8 iterations and by `6.7e-4` after 119
  iterations: the amplification of last-bit differences by the cancellation, exactly the effect the
  C-order Rust path shows against the instrumented C run. As in #358 the policy for long ill-conditioned solves is
  repeatability: the gate asserts that the device trajectory is bitwise repeatable and finite, and
  compares against the host only where the amplification has not set in (up to 2 iterations at
  `1e-8`, observed `3e-16`).

Through the `StageBackend` object: `direct_step` agrees with `StageBackend::c_order()` at `1.4e-13` (bound `2.3e-10`, kappa 9.7e2) and `cg_step` has equal iteration counts (27) and `4e-16` difference. Also gated: bitwise repeatability of the direct solve (re-upload and re-solve give identical
bits) and the CG operator with an imaginary sample matrix (`1e-15` of the largest entry).
No numerical discrepancy was found: every difference above is within its derived bound, the
only large difference (`6.7e-4`) is the CG amplification of #358 and is not a defect of either
side. Limits of the validation: the operands are synthetic (real-store structure, controlled
conditioning), not a sampled run of the optimizer, and there is one device (RTX 3060).

The C-order CPU path stays the default and the oracle; the device path never changes it.

### 15.3 Benchmark

Environment: Intel Xeon E5-2699 v3 (36 hardware threads) **shared with other jobs (load average
17 to 20 during the run)**, so the CPU all-core columns are, if anything, pessimistic for the CPU
and the 1-thread column is mildly affected; 2x NVIDIA GeForce RTX 3060 (sm_86, 12 GB, device 0),
driver 580.178.04, CUDA driver API 13.0, container CUDA toolkit 12.9.2 (cuBLAS 12.9.2, cuSOLVER
11.7.5, NVRTC 12.9), OpenBLAS 0.3.x (system, `libopenblas.so.0`), rustc 1.98.0. Medians of 3 runs
after one warm-up (one run at the two largest direct sizes). A step is what the host does today
with the C-order path (`COrderSr::gram_real` + `assemble_s_g` + `cholesky_solve`, or 50 iterations
of `SampledSrOperator::solve`, tolerance 0) with `openblas_set_num_threads(1)` and `(36)`, against
`DeviceSr`. **GPU end-to-end** includes the per-step upload of the sample matrix through pinned
staging; **GPU resident** has the matrix already on the device (what a device-side O store would
give). Synthetic problems with the structure of the real store (row 0 constant, normalized by
`1/sqrt(samples)`, latent-factor covariance), `NPara+1` rows, all components active. CSV and
metadata: `benchmark/gpu_sr_device/results/`.

**Direct SR step (Gram + S/g + Cholesky solve), milliseconds**

| NPara+1 | samples | CPU 1 thread | CPU all cores | GPU end-to-end | GPU resident | e2e vs 1 thread | e2e vs all cores | resident vs all cores | GPU phases: upload / Gram / assemble / solve (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1000 | 1000 | 60.4 | 29.1 | 21.2 | 19.7 | 2.9x | 1.4x | 1.5x | 1.5 / 10.7 / 0.1 / 8.8 |
| 1000 | 10000 | 403 | 74.0 | 120 | 108 | 3.4x | 0.6x | 0.7x | 11.9 / 99.5 / 0.1 / 8.3 |
| 3000 | 3000 | 1384 | 452 | 254 | 242 | 5.5x | 1.8x | 1.9x | 11.4 / 165 / 0.7 / 76.6 |
| 3000 | 10000 | 3270 | 709 | 658 | 626 | 5.0x | 1.1x | 1.1x | 31.9 / 548 / 0.7 / 76.6 |
| 5000 | 5000 | 5020 | 1156 | 1063 | 1038 | 4.7x | 1.1x | 1.1x | 25.6 / 741 / 2.1 / 294 |
| 10000 | 1000 | 13030 | 4164 | 2667 | 2656 | 4.9x | 1.6x | 1.6x | 10.9 / 564 / 17.3 / 2075 |
| 10000 | 10000 | 34404 | 9000 | 7815 | 7710 | 4.4x | 1.2x | 1.2x | 106 / 5609 / 17.1 / 2083 |

**CG SR step (50 CG iterations), milliseconds**

| NPara+1 | samples | CPU 1 thread | CPU all cores | GPU end-to-end | GPU resident | e2e vs 1 thread | e2e vs all cores | resident vs all cores | GPU phases: upload / Gram / assemble / solve (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1000 | 1000 | 31.2 | 13.7 | 12.2 | 10.2 | 2.5x | 1.1x | 1.3x | 1.9 / - / - / 10.1 |
| 1000 | 10000 | 856 | 292 | 42.7 | 30.8 | 20.0x | 6.8x | 9.5x | 11.9 / - / - / 30.5 |
| 3000 | 3000 | 760 | 247 | 43.3 | 28.9 | 17.6x | 5.7x | 8.5x | 11.2 / - / - / 28.8 |
| 3000 | 10000 | 2498 | 939 | 115 | 83.5 | 21.7x | 8.1x | 11.2x | 31.8 / - / - / 83.4 |
| 5000 | 5000 | 1973 | 715 | 94.5 | 69.2 | 20.9x | 7.6x | 10.3x | 25.4 / - / - / 68.9 |
| 10000 | 1000 | 729 | 200 | 49.3 | 33.7 | 14.8x | 4.1x | 5.9x | 15.6 / - / - / 33.4 |
| 10000 | 10000 | 8476 | 2754 | 382 | 275 | 22.2x | 7.2x | 10.0x | 107 / - / - / 274 |


### 15.4 When to opt in

* **CG (`NSRCG = 1`): opt in whenever `NPara * samples >= 3e6`** (for example 1000 x 10000 or
  3000 x 3000): 8.5x to 11x faster than all 36 cores resident, 5.7x to 8.1x end to end (upload
  included), 15x to 22x faster than one thread end to end (at 10^4 x 10^4: 0.38 s against 8.5 s for one thread
  and 2.75 s for 36 cores, 50 iterations). The product is memory-bandwidth bound (a 10^4 x 10^4 operand is 800 MB, read twice
  per iteration at about 300 GB/s of the card's 360), so FP64 throughput (1/64 of FP32 on this
  card) does not matter, and the per-step upload (0.1 s for 800 MB) is small against the
  iterations. Below `NPara * samples ~ 1e6` the 10 launches and 3 syncs per iteration make the
  device no faster than the CPU (1000 x 1000: 1.1x).
* **Direct SR (`NSRCG = 0`): the device is a clear win only against one core** (4.4x to 5.5x
  end to end for `NPara >= 3000`); against all 36 cores it is 1.1x to 1.8x faster at
  `NPara >= 3000` with `samples <= NPara` and at `NPara = 10^4`, and 0.6x (slower) at
  1000 x 10000. The Gram product (`NPara^2 * samples` FP64 flops, 1.6e2 GFLOP/s measured on the
  card, close to its FP64 peak) and the Cholesky (`NPara^3/3`) are compute bound, and a 36-core
  host has comparable FP64 throughput. Opt in for direct SR when the host has few cores (the
  36-core host is the strongest case for the CPU; a host with a few cores loses in proportion,
  the one-thread column is the bound), or when the O store is produced on the device. **Use CG on the device for large problems** if the physics allows
  it: it is both the larger win and the smaller memory footprint.
* **Memory:** direct needs `8 (n^2 + n_active^2 + n samples)` bytes (2.4 GB at 10^4 x 10^4, the
  largest that fits twice on a 12 GB card is about 1.5 x 10^4 x 10^4); CG needs `8 n samples`.
* **Not accelerated:** the O-store production (sampler and measurement, still on the host), the
  weight-average/MPI reductions (CG over several ranks needs the cross-rank reduction between the
  local product and the corrections, which the resident loop does not do; single-process runs
  only) and complex parameters.

### 15.5 Reproduce

```sh
MVMC_RS_CUDA_GATE=1 scripts/run_cuda_gate.sh docker            # gates, including sr_device_gate
scripts/run_sr_device_bench.sh docker                          # benchmark, writes the CSV
```

## 9. Japanese summary / 日本語要約

目的: mvmc-rs のテンソル形状の演算を、将来 GPU へ移せるように tenferro-rs 経由で表現するための
設計と棚卸し (issue #417)。コード変更はなく、C 順序の CPU 実装が引き続き数値・RNG の基準です。

* 棚卸し (Hubbard 鎖 L=16/32/64、1 スレッド、リリース版 CLI): 実行時間の大半は
  Pfaffian/逆行列 (測定側 `CalculateMAll` が 43-45 %、サンプラ側の再計算が約 16 %)、
  サンプラの rank-1 更新と比 (16-30 %)、局所エネルギー + Slater 微分 (6-10 %)。SR の S 行列・力・
  Cholesky は 0.2-0.3 % (O ベクトル保存と Gram 積を含めても 0.6-1.5 %) に過ぎず、`NPara` が
  数千以上で初めて支配的になります。
* tenferro-rs 0.7.1: バッチ付き `dot_general`/einsum、F64/C64、`cholesky`/`solve`、CUDA
  (cuBLAS/cuSOLVER/cuTENSOR)、明示的な upload/download があり、SR/CG は載せられます。
  一方、歪対称行列の LTL^T 分解・Pfaffian が無く、加算順序や決定性の保証が文書化されておらず、
  非同期転送もありません。これらを上流への機能要望として整理しました (4.7)。
* 設計: 粗い粒度 (ステージ単位) の backend trait。CPU の C 順序実装は変更せず基準として残し、
  加速実装は追加の実装として tolerance で検証します。データ配置は列優先で、バッチ次元は末尾
  (`[n,n,NQP,B]`)。複数チェーン (walker) は C の seed 規約 `RndSeed + group1` に従い、各 walker が
  ホスト側に独自の SFMT 状態を持ち、1 ステップにつきデバイスとの同期は 1 回です。1 本のチェーンだけでは
  GPU は CPU 1 コアより遅く、数百 walker 規模で初めて有利になります。
* 検証: NUMERICAL_COMPARISONS.md の絶対/相対許容誤差、教師強制リプレイ、採択判定の margin
  (`|w - draw|`) 記録、最初の乖離の特定。RNG の初期化・消費順序・回数は backend に依存せず厳密に維持します。
* ロードマップ: (1) SR/CG の pathfinder、(2) バッチ Pfaffian/逆行列、(3) walker バッチ、
  (4) 局所エネルギーのテーブル化、(5) マルチデバイス。ベンチマーク規模での Amdahl 上限は、
  Pfaffian 段階で約 2 倍、サンプラ込みで 4-9 倍、測定込みで 8-19 倍 (転送・起動コスト前)。
