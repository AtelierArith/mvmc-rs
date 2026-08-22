---
date: 2026-06-15
datetime: 2026-06-15 19:09 JST
model: GPT-5 Codex
status: plan
topic: Julia-mVMC Hubbard real-mode performance gap vs C-mVMC
---

# Julia-mVMC Hubbard real-mode performance gap plan

## Decision

Treat the Hubbard-chain Genkai profiling result as a focused performance-hardening item, separate
from the v0.4.1 MPI smoke/parity gate work.

The implementation should proceed in small, C-compatible stages.  Do not change Markov-chain
semantics, MPI sample distribution, SR solver semantics, `NSRCG` support, or `NSplitSize` support
while doing this work.

Performance comparisons must record Julia cold-JIT / first-run timing, but ratios and acceptance
targets use warm statistics that exclude run 1.

## Inputs

Genkai benchmark inputs:

- Hubbard chain, `L=16,24,32`
- `Lsub=4`
- half filling: `Ncond=L`, `2Sz=0`
- singlet spin projection matched to the Heisenberg gate:
  `NSPGaussLeg=8`, `NSPStot=0`
- `NSplitSize=1`, `NStoreO=1`, `NSRCG=0`
- MPI ranks `1,2,4`
- one software thread per rank

Warm timing source:

- `/home/pj25000164/ku50001833/shin-mvmc-v04-mpi-benchmark/runs/v04-mpi-hubbard-warm-timing-c-v14-20260615-150924`

Julia timer/profile source:

- `/home/pj25000164/ku50001833/shin-mvmc-v04-mpi-benchmark/runs/v04-mpi-hubbard-profile-lscale-julia-20260615-152303`
- `/home/pj25000164/ku50001833/shin-mvmc-v04-mpi-benchmark/runs/v04-mpi-hubbard-l32-line-profile-20260615-183652`

Baseline:

- Primary correctness baseline: C-mVMC v1.4.0 `icc` and `gcc`
- Performance reference for the current investigation: C-mVMC v1.4.0 `gcc`

## Observed Gaps

The pronounced C/Julia gaps for Hubbard real mode are not mainly in `CalculateMAll`.
At `L=32`, `CalculateMAll` is only about `1.1x` slower in Julia than in C `gcc`.

The larger differences are:

| Timer | Example | C `gcc` warm median | Julia warm/profile median | Julia/C |
|---|---:|---:|---:|---:|
| `CalHamiltonian1` | `L=32`, rank 4 | `0.148s` | `0.624s` | `4.2x` |
| `multiply store OO` | `L=32`, rank 4 | `0.0505s` | `0.3469s` | `6.9x` |
| `WeightAverage` | `L=32`, rank 4 | `0.032s` | `0.452s` | `14x` |

The `WeightAverage` ratio is sensitive to MPI noise and profile overhead, but the implementation
difference is real enough to fix.

## Scope

In scope:

- Real-mode Hubbard path under `NSRCG=0`, `NStore=1`, `NSplitSize=1`
- `VMCParaOpt`
- MPI ranks `1,2,4`
- one software thread per rank
- C-compatible timing labels and benchmark reporting

Out of scope:

- `NSRCG != 0` implementation or behavior changes
- `NSplitSize > 1`
- independent-chain sampling
- outer-loop sampling threading
- broad complex-mode performance work, except where a small shared helper can be made safely

## Workstream 0: Preserve Baseline And Gates

Goal: keep the existing correctness and timing evidence reproducible before changing kernels.

Tasks:

- Save the current Genkai Hubbard timing/profile summary into a report or append it to the
  relevant benchmark report.
- Keep `cold_s` in timing tables, but use `warm_median_s` for ratios.
- Keep C `gcc` and C `icc` outputs in the same correctness table.
- Add or keep a small local test that compares the first 10 `zvo_out` rows for Hubbard `L=16`
  against C before and after each optimization stage.
- Define and record parity tolerance before changing kernels:
  - `parity pass` means the first 10 `zvo_out` rows pass the existing numeric C/Julia
    comparator against both C builds.
  - Workstreams 1 and 3 may change floating summation or MPI reduction order, so do not require
    bitwise or text equality there.  Record the comparator tolerance and the maximum observed
    absolute/relative difference.
  - If no existing tolerance is recorded in the benchmark script, start with `atol=1e-10` and
    `rtol=1e-10`, then tighten or relax only with an explicit note in the benchmark report.
  - Workstream 2 should preserve the local arithmetic order more closely; a prefix mismatch beyond
    the documented tolerance is a blocker.

Acceptance:

- Current `L=16,24,32`, ranks `1,2,4` first-10-step Hubbard parity remains pass against both C
  builds after each stage.
- Timing output clearly distinguishes cold timing from warm timing.

## Workstream 1: Replace Real `OO` Store Loop With BLAS

Goal: make Julia's `multiply store OO` match C-mVMC's algorithmic path.

Current Julia path:

- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl:2648`
- `finalize_oo_store_real!` uses a scalar triple loop for `NSRCG=false`.

C reference:

- `mVMC/src/mVMC/vmccal.c:660`
- `calculateOO_Store_real` calls `M_DGEMM` when `NSRCG==0`.

Plan:

- For real `NSRCG=false`, reshape `sr_opt_o_store_real` as
  `(sr_opt_size, sample_size)` without copying.
- Use contiguous strided views for both operands and output, e.g.
  `reshape(@view(sr_opt_o_store_real[1:sr_opt_size*sample_size]), sr_opt_size, sample_size)` and
  `reshape(@view(sr_opt_oo_real[1:sr_opt_size*sr_opt_size]), sr_opt_size, sr_opt_size)`.
- Compute `sr_opt_oo_real[1:sr_opt_size^2] = O * transpose(O)` using BLAS-backed
  `mul!` or `LinearAlgebra.BLAS.gemm!`.
- Preserve C layout: leading dimension is `sr_opt_size`, and active `OO` occupies the first
  `sr_opt_size * sr_opt_size` entries.
- Keep C fidelity by computing the full matrix with GEMM.  Do not switch to `SYRK` or a
  triangular-only update in this workstream.
- Leave the `NSRCG=true` reduced storage path unchanged in this workstream.
- Add a focused unit test comparing the previous scalar-loop result and the BLAS result for
  deterministic small matrices, including non-square `sample_size`.  This comparison must be
  tolerance-based, not bitwise.

Risks:

- A stray slice copy would hide the intended speedup.
- A non-strided view or accidental copy could make `mul!` use a generic fallback instead of BLAS;
  verify dispatch and allocation behavior locally.
- BLAS thread oversubscription would distort MPI timings; keep `OPENBLAS_NUM_THREADS=1` and
  `MKL_NUM_THREADS=1` in benchmarks.
- This replaces the current Julia `@threads` store kernel with BLAS in the `NSRCG=false` real path.
  That is aligned with the one-thread-per-rank scope and C's current algorithm.  Future BLAS-thread
  versus Julia-thread policy should be handled separately.
- Floating summation order changes may slightly change later SR trajectory, but C already uses
  BLAS here, so this moves Julia closer to C.

Acceptance:

- `multiply store OO` for Hubbard `L=32` falls from about `0.35s` toward the C range
  (`~0.05s`; initial target `<=0.10s`).
- Hubbard `L=16` first-10-step C/Julia parity remains pass under the Workstream 0 documented
  tolerance.
- Existing Heisenberg MPI gate remains pass.

## Workstream 2: Add Allocation-Free Real `green_func1` Path

Goal: reduce the `CalHamiltonian1` gap by removing per-transfer allocations and matching C's
workspace reuse style.

Current Julia path:

- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl:97`
- `green_func1` copies `ele_idx` and `ele_num`, allocates `proj_cnt_new`, and allocates
  `pf_m_new_real` for each one-body transfer.
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl:1217`
- `CalHamiltonian1` loops over `data.transfer_terms`, branches on `term.spin`, and calls the
  generic `green_func1` path.

C reference:

- `mVMC/src/mVMC/calham_real.c:129`
- `mVMC/src/mVMC/locgrn_real.c:44`
- `GreenFunc1_real` mutates `eleIdx` and `eleNum`, uses caller-provided `projCntNew` and
  `buffer`, then reverts the mutation before returning.

Plan:

- Introduce a real-mode scratch workspace for the main-calculation path:
  `proj_cnt_new`, `pf_m_new_real`, and any temporary electron-index / electron-number buffers
  needed to preserve safety.
- Keep the scratch boundary compatible with C's later `GreenFunc2_real` reuse pattern:
  `GreenFunc1_real` needs `NQPFull`, while `GreenFunc2_real` needs `NQPFull + 2*Nsize`.
  This workstream only optimizes `CalHamiltonian1`; `CalHamiltonian2` may remain on the existing
  generic path unless it is explicitly staged later.
- Implement an internal `green_func1_real!` that:
  - handles early exits before mutation;
  - mutates the active buffers in the same order as C;
  - calls `update_proj_cnt!`, `proj_rbm_ratio`, `calculate_new_pf_m2_real!`, and
    `calculate_ip_real`;
  - reverts the electron move before returning;
  - performs no heap allocation in the hot path.
- Keep the generic `green_func1` available for complex mode and existing callers.
- Consider pre-expanding real transfer terms into spin-resolved integer/value arrays so the hot
  loop avoids `Symbol` checks and `ComplexF64` arithmetic when `all_complex=false`.

Risks:

- Mutation/revert bugs can create trajectory divergence.  The implementation must be staged and
  checked with strict short-trajectory parity.
- Shared workspace must remain rank-local and thread-safe.  Since this scope is one software
  thread per rank, do not prematurely design a threaded workspace.
- Projection/RBM ratio behavior must remain identical to the current Julia path and C path.

Acceptance:

- Allocation profiling for `CalHamiltonian1` shows no per-transfer `copy` / `zeros` allocation
  in the real Hubbard path.
- `CalHamiltonian1` for Hubbard `L=32` improves materially from the current `~4.2x` C `gcc`
  ratio; initial target `<=2x`.
- Hubbard and Heisenberg strict prefix parity remains pass.

## Workstream 3: Make Real `WeightAverage` Collective C-Compatible

Goal: reduce MPI overhead and active data volume in `WeightAverageSROpt_real`.

Current Julia path:

- `Julia-mVMC/MVMCOptimizers.jl/src/weight_average.jl:133`
- Julia calls `allreduce_sum!` separately for `sr_opt_oo_real` and `sr_opt_ho_real`.
- Julia calls MPI allreduce whenever `ctx.is_mpi=true`, including `mpiexec -n 1` runs.
- `Julia-mVMC/MVMCOptimizers.jl/src/types.jl:65`
- `sr_opt_oo_real` is allocated as `sr_opt_size * (sr_opt_size + 2)`, while `sr_opt_ho_real`
  is a separate vector.

C reference:

- `mVMC/src/mVMC/average.c:115`
- For `NSRCG==0`, C reduces `SROptSize * (SROptSize + 1)` doubles from the contiguous
  `SROptOO_real` block, covering `OO + HO` and excluding `SROptO`.
- C skips `SafeMpiAllReduce` when communicator size is `1`.

Plan:

- First diagnose the current `WeightAverage` cost before changing storage:
  - record `sr_opt_size`, active `OO` length, `HO` length, current reduced lengths, and chunk count;
  - time the `OO` and `HO` allreduce calls separately for ranks `1,2,4`;
  - repeat warm measurements enough to classify the cost as latency-dominated, volume-dominated,
    or profiler/MPI-noise dominated.
- Add the C-compatible `size==1` fast path: when `ctx.is_mpi=true` but `ctx.size0 == 1`, skip
  MPI allreduce and only normalize local active data.
- Then reduce only the active real `OO` range instead of the full allocated `sr_opt_oo_real`
  buffer.
- After the diagnosis, evaluate combining `OO` and `HO` into one collective:
  - either by making a C-compatible contiguous active buffer;
  - or by packing `OO + HO` into a reusable communication buffer and unpacking after one
    Allreduce.
- Keep normalization by `1 / Wc` consistent with C.
- Preserve `NSRCG=true` behavior as a separate later audit; this plan targets `NSRCG=false`.

Risks:

- Changing storage layout touches SR solver assumptions in `stochastic_opt.jl`; prefer a minimal
  active-view or reusable-pack-buffer change first.
- Packing into a fresh buffer every step can erase the benefit of one collective; any pack buffer
  must be reusable and included in allocation checks.
- For small buffers, measured MPI time can be noisy.  Use repeated warm measurements and compare
  ranks `2` and `4`.
- If diagnosis shows volume-dominated behavior, one collective alone is unlikely to explain a
  large improvement; record that result before changing the implementation.

Acceptance:

- A diagnostic note records message sizes, call counts, per-call timing, and the inferred
  latency/volume/noise classification.
- `mpiexec -n 1` does not call MPI allreduce in `WeightAverageSROpt_real`.
- Julia uses one collective for the `OO + HO` active real SR data, or the plan records why this is
  not safe yet.
- `WeightAverage` rank-4 timing no longer dominates the C/Julia gap for Hubbard `L=32`.
- Correctness remains pass for first-10-step Hubbard parity under the Workstream 0 documented
  tolerance.

## Verification Matrix

After each workstream:

- local Julia unit tests for the touched helper
- existing MPI smoke tests, including Hubbard `mpiexec -n 2`
- Genkai short Hubbard parity:
  - `L=16`, ranks `1,2,4`
  - C `icc` and C `gcc`
  - first 10 `zvo_out` rows
  - comparator tolerance and maximum absolute/relative difference recorded

After Workstreams 1-3:

- Genkai warm timing, cold run recorded but excluded from ratios:
  - `L=16,24,32`
  - ranks `1,2,4`
  - C `gcc`, C `icc`, Julia
  - repeats `>=3`
- Julia timer profile with `MVMC_C_TIMER=1`.
- Compare `CalHamiltonian1`, `multiply store OO`, `WeightAverage`, `VMCMainCal`, and total
  warm time.

## Suggested Implementation Order

1. Land/report the baseline evidence and keep the benchmark scripts stable.
2. Implement Workstream 1 (`OO` BLAS) first because it is local, C-matched, and high-confidence.
3. Re-run Hubbard `L=16` parity and one `L=32` timing/profile check.
4. Implement Workstream 2 (`green_func1_real!` workspace) in small commits.
5. Re-run documented-tolerance parity before broad timing.
6. Diagnose Workstream 3 (`WeightAverage`) call count/message size/rank scaling before changing
   collective layout.
7. Implement Workstream 3 after confirming the diagnosis and SR storage layout.
8. Run the full Genkai warm timing matrix and update the benchmark report.

Remote operations such as push, PR creation, and merge require explicit user confirmation before
each operation.
