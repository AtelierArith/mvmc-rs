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
benchmark metadata block. It is backend-agnostic through the trait `AcceleratedStages` (a stage
that is not implemented reports `Unsupported`, never a CPU fallback). The CPU tenferro variant
runs in normal CI; the CUDA variant is the second ignored test of the CUDA gate (CUDA RTX 3060
run: S and g match the oracle, 0 flips, 0 defects, Pfaffian stage unsupported). A later batched
Pfaffian implementation (#423) plugs in by implementing `pfaffian_inverse`. The policy text is
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
