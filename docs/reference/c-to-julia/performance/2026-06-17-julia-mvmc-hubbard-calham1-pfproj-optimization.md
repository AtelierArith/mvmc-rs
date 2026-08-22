---
date: 2026-06-17
datetime: 2026-06-17 11:34 JST
model: GPT-5 Codex
status: report
topic: Julia-mVMC Hubbard CalHamiltonian1 Pfaffian/projector optimization on Genkai
related:
  - docs/reports/2026-06-15-julia-mvmc-hubbard-ws123-performance-diagnostics.md
  - docs/reports/2026-06-17-julia-mvmc-v041-ohtaka-genkai-runtime-comparison.md
  - docs/plans/2026-06-15-julia-mvmc-hubbard-performance-gap-plan.md
---

# Julia-mVMC Hubbard CalHamiltonian1 Pfaffian/projector optimization

## Summary

2026-06-17 JST に、Genkai の Hubbard real-mode で Julia-mVMC が C-mVMC より遅い主因を `CalHamiltonian1 [71]` まで掘り下げ、`green_func1_real!` 内の Pfaffian 更新と projector 更新を最適化した。

最終的な有効版は `pfproj3` である。

- `CalHamiltonian1 [71]` は、直前の workspace 版に対して `0.69x` から `0.74x` まで短縮した。
- C `gcc` に対する `CalHamiltonian1` の差は、最終的に `1.10x` から `1.18x` 程度まで縮まった。
- C `icc` に対してはまだ `1.39x` から `1.54x` 程度の差が残る。
- `zvo_CalcTimer.dat` の `All [0]` 全体では、Julia は C `gcc` 比で `1.06x` から `1.31x`、C `icc` 比で `0.96x` から `1.21x` の範囲になった。
- 一度、診断 timer 920-924 が通常 `MVMC_C_TIMER=1` にも混入して `CalHamiltonian1` を悪化させる問題を見つけた。最終版では `MVMC_CALHAM1_DIAG=1` の時だけ診断 timer を有効にするよう分離した。

今回の重要な教訓は、細粒度 timer は hot path では容易に測定対象を壊すため、通常 CTimer と明確に分離する必要がある、という点である。

## Scope

対象:

- Problem: Hubbard chain, real mode
- Sizes: `L=16,24,32`
- MPI ranks: `R=4`
- Repeats: `3`
- Warm median: `run_02` / `run_03` の median
- Threads:
  - `JULIA_NUM_THREADS=1`
  - `OMP_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
- Platform: Genkai, PJM 1 node
- Input source:
  - `/home/pj25000164/ku50001833/shin-mvmc-v04-mpi-benchmark/runs/v04-mpi-hubbard-warm-timing-c-v14-20260615-150924`

主に見た CTimer section:

- `All [0]`
- `VMCMainCal [4]`
- `LocEnergyCal [41]`
- `CalHamiltonian1 [71]`
- `ReturnSlaterElmDiff [42]`

## Pre-Optimization State

前段階では、`CalHamiltonian1 [71]` は C-mVMC より明確に遅かった。

Baseline source/result:

- Source:
  - `/home/pj25000164/ku50001833/shin-mvmc-v04-mpi-benchmark/source/Julia-mVMC-workspace-calham1-slater-ws-20260617-103259`
- Local recovered result:
  - `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-slater-ws-ctimer-R4-20260617-103259-retry1`

`CalHamiltonian1 [71]`, warm median `run_02/run_03`:

| L | Julia before | C icc | Julia/C icc | C gcc | Julia/C gcc |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.07919 | 0.03810 | 2.078 | 0.05328 | 1.486 |
| 24 | 0.14871 | 0.07371 | 2.018 | 0.09142 | 1.627 |
| 32 | 0.24218 | 0.11464 | 2.112 | 0.14719 | 1.645 |

## Why Hubbard Was Different From Heisenberg

`CalHamiltonian1 [71]` は one-body transfer term を処理する section である。

Julia 側では `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl` の `calculate_hamiltonian` 内で、`data.transfer_terms` を走査し、real mode では `green_func1_real!` を呼ぶ。

Hubbard では hopping/transfer が直接この経路を大量に踏む。一方、Heisenberg は主に exchange/two-body 側の `CalHamiltonian2 [72]` に寄るため、Hubbard ほど `green_func1_real!` の単発 hopping 経路が支配的にならない。

## Diagnostic Timer

`CalHamiltonian1` の内訳を切るため、診断 timer を追加した。

追加 section:

| id | name |
|---:|---|
| 920 | `CalH1 GreenFunc1Real` |
| 921 | `UpdateProjCnt` |
| 922 | `ProjRatio` |
| 923 | `CalculateNewPfM2_real` |
| 924 | `CalculateIP_real` |

対象ファイル:

- `Julia-mVMC/MVMCOptimizers.jl/src/c_timer.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/run_para_opt_from_namelist.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl`

診断結果の回収先:

- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-diag-R4-20260617-110124`
- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-diag-L24-R4-20260617-110259`
- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-diag-L32-R4-20260617-110300`

診断結果、warm median `run_02/run_03`:

| L | GreenFunc1 / CalH1 | UpdateProjCnt | ProjRatio | CalculateNewPfM2 | CalculateIP | other |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 86.6% | 18.2% | 13.6% | 27.3% | 7.4% | 33.4% |
| 24 | 88.3% | 19.9% | 14.4% | 31.3% | 6.2% | 28.2% |
| 32 | 89.3% | 20.9% | 14.7% | 33.7% | 6.1% | 24.7% |

注意:

- この診断 run では細粒度 timer の overhead により `CalHamiltonian1 [71]` 自体が `1.5x` から `1.7x` 程度に膨らんだ。
- したがって、絶対時間ではなく構成比を見る用途に限定した。
- 最初の診断実装では 920-924 が通常 `MVMC_C_TIMER=1` でも有効になっていた。最終版では `MVMC_CALHAM1_DIAG=1` の時だけ有効化する。

## Implementation Changes

### 1. `calculate_new_pf_m2_real!`

対象:

- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl`

変更点:

- C-mVMC の `mVMC/src/mVMC/pfupdate_real.c` の `CalculateNewPfM_real` / `CalculateNewPfM2_real` に合わせ、内側 loop を spin up/down の 2 本に分割した。
- 変更前は hot loop 内に `msj < n_elec` 分岐があり、各 iteration で `rsj` を分岐計算していた。
- 変更後は、
  - `msj = 1:n_elec`
  - `msj = n_elec+1:n_size`
  の 2 loop に分け、down-spin 側だけ `+ n_site` する形にした。
- public 関数としての bounds check は残しつつ、`green_func1_real!` からは `@inbounds calculate_new_pf_m2_real!` で呼び、hot path では check を elide できる形にした。

最終形で残した判断:

- `@simd` は最終版では使わない。
- 一度 `@simd` 付き split loop を試したが、Genkai の `CalHamiltonian1` は悪化した。
- C 側には `#pragma omp simd reduction(+:ratio)` があるが、Julia 側で同じ annotation が常に良い結果になるわけではなかった。

### 2. `UpdateProjCnt`

対象:

- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl`

変更点:

- `ele_num` の up/down `@view` 作成をやめ、`ele_num[ri+1]` と `ele_num[n_site+ri+1]` を直接参照する形にした。
- Jastrow の対称 index 取得を local nested function から `@inline _symmetric_site_pair_idx` に出した。
- Gutzwiller/Jastrow の更新式は C と同じ意味を保った。
- DH2/DH4 tail は既存の `_recompute_dh_counts!` を維持した。

### 3. `ProjRatio`

対象:

- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl`

変更点:

- `log_proj_ratio_noalloc` で、各 projection section の有効 loop 長を先に計算し、loop 内の `idx <= length(...)` 判定を外した。
- `@simd` は使わず、通常の順序和を維持した。C 側の `ProjRatio` も通常 loop で、reduction pragma はない。

### 4. Diagnostic Timer Gating

対象:

- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/run_para_opt_from_namelist.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/c_timer.jl`

問題:

- 最初の診断 timer 実装では、`MVMC_C_TIMER=1` の通常 CTimer run でも 920-924 の細粒度 timer が動いていた。
- これは `CalHamiltonian1 [71]` の hot path に timer overhead を混入させ、最適化結果を悪化して見せた。

修正:

- `vmc_main_cal!` 内で `MVMC_CALHAM1_DIAG` を読み、診断用には `calham1_diag_timer` を別に流す。
- `MVMC_CALHAM1_DIAG=0` の通常 run では `calham1_diag_timer = CTIMER_DISABLED` になり、920-924 は no-op になる。
- 通常 CTimer の 70/71/72 等は従来通り `c_timer` で取る。

## Runs

### Diagnostic runs

| Purpose | Job | Local recovered result |
|---|---:|---|
| CalH1 diagnostic L16/R4 | 6055015 | `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-diag-R4-20260617-110124` |
| CalH1 diagnostic L24/R4 | 6055021 | `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-diag-L24-R4-20260617-110259` |
| CalH1 diagnostic L32/R4 | 6055022 | `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-diag-L32-R4-20260617-110300` |

### Failed/diagnostic-only optimization runs

These runs are important as negative results.

| Label | Jobs | Local recovered result | Outcome |
|---|---|---|---|
| `pfproj` | 6055390, 6055392, 6055393 | `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-pfproj-opt-*` | Bad: `@simd` split plus diagnostic timer overhead made `CalHamiltonian1` worse |
| `pfproj2` | 6055454, 6055456, 6055457 | `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-pfproj2-opt-*` | Still bad because 920-924 diagnostic timer was active during normal `MVMC_C_TIMER=1` |

### Final run

Final source:

- `/home/pj25000164/ku50001833/shin-mvmc-v04-mpi-benchmark/source/Julia-mVMC-calham1-pfproj3-opt-20260617-112403`

Jobs:

| L | Job | Local recovered result |
|---:|---:|---|
| 16 | 6055480 | `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-pfproj3-opt-L16-R4-20260617-112421` |
| 24 | 6055481 | `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-pfproj3-opt-L24-R4-20260617-112422` |
| 32 | 6055483 | `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-pfproj3-opt-L32-R4-20260617-112424` |

## Results

### `CalHamiltonian1 [71]`: Before vs After

Warm median `run_02/run_03`.

| L | before | after | after/before |
|---:|---:|---:|---:|
| 16 | 0.07919 | 0.05853 | 0.739 |
| 24 | 0.14871 | 0.10272 | 0.691 |
| 32 | 0.24218 | 0.17407 | 0.719 |

### `CalHamiltonian1 [71]`: Final Julia vs C-mVMC

Warm median `run_02/run_03`.

| L | Julia after | C icc | Julia/C icc | C gcc | Julia/C gcc |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.05853 | 0.03810 | 1.536 | 0.05328 | 1.099 |
| 24 | 0.10272 | 0.07371 | 1.394 | 0.09142 | 1.124 |
| 32 | 0.17407 | 0.11464 | 1.518 | 0.14719 | 1.183 |

### Overall `All [0]`: Final Julia vs C-mVMC

Warm median `run_02/run_03`.

| L | Julia | C icc | Julia/C icc | C gcc | Julia/C gcc |
|---:|---:|---:|---:|---:|---:|
| 16 | 1.02811 | 1.07297 | 0.958 | 0.97034 | 1.060 |
| 24 | 2.63467 | 2.17668 | 1.210 | 2.00459 | 1.314 |
| 32 | 3.83184 | 3.61216 | 1.061 | 3.48688 | 1.099 |

Interpretation:

- `L=16`: 全体は C `icc` より少し速く、C `gcc` より約 `6%` 遅い。
- `L=32`: 全体は C `icc` 比 `1.06x`、C `gcc` 比 `1.10x` でかなり近い。
- `L=24`: 全体は C `icc` 比 `1.21x`、C `gcc` 比 `1.31x` でまだ差が大きい。

### Key Sections

Warm median `run_02/run_03`.

| L | section | Julia | C icc | Julia/C icc | C gcc | Julia/C gcc |
|---:|---|---:|---:|---:|---:|---:|
| 16 | `VMCMainCal [4]` | 0.57132 | 0.55347 | 1.032 | 0.55209 | 1.035 |
| 16 | `LocEnergyCal [41]` | 0.06094 | 0.04394 | 1.387 | 0.06044 | 1.008 |
| 16 | `ReturnSlaterElmDiff [42]` | 0.09665 | 0.04727 | 2.044 | 0.10808 | 0.894 |
| 24 | `VMCMainCal [4]` | 1.23069 | 1.10006 | 1.119 | 1.13051 | 1.089 |
| 24 | `LocEnergyCal [41]` | 0.10508 | 0.07992 | 1.315 | 0.09875 | 1.064 |
| 24 | `ReturnSlaterElmDiff [42]` | 0.18576 | 0.09352 | 1.986 | 0.20059 | 0.926 |
| 32 | `VMCMainCal [4]` | 2.07769 | 1.82360 | 1.139 | 1.93654 | 1.073 |
| 32 | `LocEnergyCal [41]` | 0.17673 | 0.12094 | 1.461 | 0.15511 | 1.139 |
| 32 | `ReturnSlaterElmDiff [42]` | 0.30136 | 0.14856 | 2.029 | 0.32648 | 0.923 |

`ReturnSlaterElmDiff [42]` は C `gcc` には勝っているが、C `icc` には約 `2x` 遅い。次に詰めるなら C `icc` との差は Slater derivative 側も見る価値がある。

## Verification

Local tests run after final changes:

- `julia --project=. test_unit/test_unit_vmc_main_cal_sr.jl`
- `julia --project=. test_unit/test_unit_vmc_sampling_proj.jl`
- `julia --project=. test_unit/test_unit_threading.jl`
- `julia --project=. test/runtests.jl`
- `git diff --check`

All passed.

## Current Modified Files

At the time of this report, the relevant working tree changes are:

- `Julia-mVMC/MVMCOptimizers.jl/src/c_timer.jl`
  - CalH1 diagnostic timer output definition.
- `Julia-mVMC/MVMCOptimizers.jl/src/run_para_opt_from_namelist.jl`
  - `MVMC_CALHAM1_DIAG` handling and diagnostic output.
- `Julia-mVMC/MVMCOptimizers.jl/src/threading.jl`
  - Main-cal scratch buffers from the earlier workspace optimization.
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl`
  - `green_func1_real!` scratch/timer path.
  - `log_proj_ratio_noalloc` loop-bound cleanup.
  - sample buffer reuse and Slater workspace from the earlier optimization.
  - diagnostic timer gating.
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl`
  - `update_proj_cnt!` cleanup.
  - `calculate_new_pf_m2_real!` split-loop optimization.
- `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_vmc_main_cal_sr.jl`
  - scratch path tests.

## Follow-Up

Recommended next steps:

1. Keep the diagnostic timer gating as-is. Do not let 920-924 run during normal `MVMC_C_TIMER=1`.
2. If further CalHamiltonian1 work is needed, focus on residual C `icc` gap:
   - `calculate_ip_real` loop overhead is small but still measurable.
   - `UpdateProjCnt` / `ProjRatio` now improved structurally, but DH tail recompute cost may matter for inputs with DH correlations.
3. For whole-program C `icc` parity, inspect `ReturnSlaterElmDiff [42]` next. It remains about `2x` slower than C `icc`, although it is already faster than C `gcc`.
4. Repeat a correctness gate against C output before merging these performance changes into release-facing branches.
