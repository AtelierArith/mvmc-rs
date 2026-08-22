---
date: 2026-06-17
datetime: 2026-06-17 14:24 JST
model: GPT-5 Codex
status: report
topic: Julia-mVMC Hubbard LocEnergy/Slater/CalHamiltonian1 optimization record on Genkai
related:
  - docs/reports/2026-06-17-julia-mvmc-hubbard-optimization-final-summary.md
  - docs/reports/2026-06-15-julia-mvmc-hubbard-ws123-performance-diagnostics.md
  - docs/reports/2026-06-17-julia-mvmc-hubbard-calham1-pfproj-optimization.md
  - docs/reports/2026-06-17-julia-mvmc-v041-ohtaka-genkai-runtime-comparison.md
---

# Julia-mVMC Hubbard LocEnergy/Slater/CalHamiltonian1 optimization record

## Summary

2026-06-17 JST に、Genkai の Hubbard real-mode, ranks `4` で Julia-mVMC の
`LocEnergyCal [41]` / `CalHamiltonian1 [71]` / `ReturnSlaterElmDiff [42]`
を C-mVMC `gcc` / `icc` と比較しながら最適化した。

この文書は、同日の中間記録
`docs/reports/2026-06-17-julia-mvmc-hubbard-calham1-pfproj-optimization.md`
を更新・統合する最終記録である。中間記録の `pfproj3` 後に、Slater fast
path、CalHamiltonian1 outer fast path、direct projection ratio、PfM2/IP
fused、projection value cache を追加し、最終的に `CalHamiltonian1 [71]`
は C `icc` 比 `1.07x` から `1.14x` まで縮まった。

最終版の要点:

- `ReturnSlaterElmDiff [42]` は C `icc` とほぼ同等になった。
- `CalHamiltonian1 [71]` は旧 Slater-fast 版から `0.74x` から `0.77x`
  程度まで短縮した。
- `CalHamiltonian1 [71]` は C `gcc` より速く、C `icc` との差は
  `7%` から `14%` 程度まで縮まった。
- `LocEnergyCal [41]` は `CalHamiltonian1 [71]` にほぼ追随し、C `icc`
  比 `1.02x` から `1.08x` 程度まで縮まった。
- `All [0]` や `VMCMainCal [4]` は run-to-run noise と collective wait の
  影響を受けるため、kernel 改善の評価には warm median と section timer を
  併用する必要がある。

## Scope

対象:

- Problem: Hubbard chain, real mode
- Sizes: `L=16,24,32`
- MPI ranks: `R=4`
- Repeats: `3`
- Warm median: `run_02` / `run_03` の median
- Genkai PJM 1 node
- Threads:
  - `JULIA_NUM_THREADS=1`
  - `OMP_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
- Input source:
  - `/home/pj25000164/ku50001833/shin-mvmc-v04-mpi-benchmark/runs/v04-mpi-hubbard-warm-timing-c-v14-20260615-150924`

主に見た CTimer section:

- `All [0]`
- `VMCMainCal [4]`
- `CalculateMAll [40]`
- `LocEnergyCal [41]`
- `CalHamiltonian1 [71]`
- `ReturnSlaterElmDiff [42]`
- `calculate OO and HO [43]`

## Final Results

### Normal CTimer, warm median seconds

`run_02` / `run_03` の median。`Julia projcache` が最終版。

| L | series | All[0] | VMCMainCal[4] | CalculateMAll[40] | LocEnergy[41] | CalH1[71] | Slater[42] | OOHO[43] |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | Julia slater-fast | 1.31759 | 0.51157 | 0.37472 | 0.05868 | 0.05641 | 0.04613 | 0.00353 |
| 16 | Julia outer-fast | 1.08174 | 0.51967 | 0.38085 | 0.05727 | 0.05505 | 0.04506 | 0.00366 |
| 16 | Julia direct-proj | 1.07624 | 0.51250 | 0.38126 | 0.04812 | 0.04582 | 0.04629 | 0.00351 |
| 16 | Julia fused | 1.07763 | 0.50009 | 0.37584 | 0.04677 | 0.04457 | 0.04505 | 0.00358 |
| 16 | Julia projcache | 1.07102 | 0.50167 | 0.37433 | 0.04584 | 0.04358 | 0.04581 | 0.00348 |
| 16 | C gcc | 0.97034 | 0.55209 | 0.35979 | 0.06044 | 0.05328 | 0.10808 | 0.00380 |
| 16 | C icc | 1.07297 | 0.55347 | 0.45212 | 0.04394 | 0.03810 | 0.04727 | 0.00321 |
| 24 | Julia slater-fast | 2.71076 | 1.14128 | 0.89343 | 0.10477 | 0.10242 | 0.08846 | 0.00413 |
| 24 | Julia outer-fast | 2.58502 | 1.16856 | 0.90217 | 0.10567 | 0.10328 | 0.08759 | 0.00468 |
| 24 | Julia direct-proj | 2.56311 | 1.11681 | 0.89192 | 0.08550 | 0.08310 | 0.08733 | 0.00458 |
| 24 | Julia fused | 2.46367 | 1.16212 | 0.90570 | 0.08524 | 0.08288 | 0.08805 | 0.00468 |
| 24 | Julia projcache | 2.36711 | 1.22075 | 0.90156 | 0.08127 | 0.07883 | 0.08709 | 0.00436 |
| 24 | C gcc | 2.00459 | 1.13051 | 0.78984 | 0.09875 | 0.09142 | 0.20059 | 0.00473 |
| 24 | C icc | 2.17668 | 1.10006 | 0.91081 | 0.07992 | 0.07371 | 0.09352 | 0.00405 |
| 32 | Julia slater-fast | 3.85637 | 1.92791 | 1.51171 | 0.17464 | 0.17218 | 0.14699 | 0.00508 |
| 32 | Julia outer-fast | 3.74813 | 1.85733 | 1.46589 | 0.17300 | 0.17048 | 0.14660 | 0.00524 |
| 32 | Julia direct-proj | 3.86035 | 1.83228 | 1.48127 | 0.13600 | 0.13340 | 0.14588 | 0.00519 |
| 32 | Julia fused | 3.72606 | 1.84139 | 1.48924 | 0.13422 | 0.13165 | 0.14583 | 0.00516 |
| 32 | Julia projcache | 3.70648 | 1.86543 | 1.51321 | 0.13048 | 0.12789 | 0.14830 | 0.00525 |
| 32 | C gcc | 3.48688 | 1.93654 | 1.38867 | 0.15511 | 0.14719 | 0.32648 | 0.00616 |
| 32 | C icc | 3.61216 | 1.82360 | 1.52704 | 0.12094 | 0.11464 | 0.14856 | 0.00524 |

### CalHamiltonian1 ratios

| L | projcache / slater-fast | projcache / outer-fast | projcache / fused | projcache / C icc | projcache / C gcc |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.773 | 0.792 | 0.978 | 1.144 | 0.818 |
| 24 | 0.770 | 0.763 | 0.951 | 1.069 | 0.862 |
| 32 | 0.743 | 0.750 | 0.972 | 1.116 | 0.869 |

Interpretation:

- `CalHamiltonian1 [71]` は C `gcc` より `13%` から `18%` 速い。
- C `icc` に対してはまだ `7%` から `14%` 遅い。
- `L=24` が最も近く、`CalHamiltonian1 [71]` は C `icc` 比 `1.069x`。

## Optimization Timeline

### 1. Slater fast path

対象:

- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/threading.jl`

内容:

- `slater_elm_diff_fcmp!` に scratch path を追加した。
- `transOrb build` と `buffer accumulate` で、hot loop 内の bounds/fallback を外す fast path を入れた。
- `srOptO` store も scratch buffer 経由で安定化した。

結果:

- `ReturnSlaterElmDiff [42]` は C `icc` とほぼ同等。
- warm median:
  - `L16`: Julia `0.04613`, C `icc` `0.04727`
  - `L24`: Julia `0.08846`, C `icc` `0.09352`
  - `L32`: Julia `0.14699`, C `icc` `0.14856`

### 2. CalHamiltonian1 outer fast path

対象:

- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/threading.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/c_timer.jl`

内容:

- `calculate_hamiltonian1_real_fast!` を追加した。
- `data.transfer_terms` を `(ri,rj,spin,value)` の scratch arrays に展開した。
- `term.spin == :up/:down/:both` 分岐を hot path から外した。
- real/no-RBM/scratch 条件でのみ fast path を使い、複素 transfer value などは generic path に fallback する。
- `JULIA_MVMC_INNER_THREADS=1` と `allow_inner_threads=true` の時だけ、Transfer loop の gated threading を使う分岐を用意した。

結果:

- outer loop の最適化だけでは `CalHamiltonian1 [71]` の絶対値はほぼ変わらなかった。
- ここで、支配的なのは Transfer loop 本体ではなく `GreenFunc1_real` 内部だと確認した。

### 3. CalHamiltonian1 diagnostic timer

対象:

- `Julia-mVMC/MVMCOptimizers.jl/src/c_timer.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/run_para_opt_from_namelist.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl`

追加した id:

| id | label |
|---:|---|
| 920 | `CalH1 GreenFunc1Real` |
| 921 | `UpdateProjCnt` |
| 922 | `ProjRatio` |
| 923 | `CalculateNewPfM2_real` |
| 924 | `CalculateIP_real` |
| 925 | `CalH1 fast prep` |
| 926 | `CalH1 fast term loop` |
| 927 | `GreenFunc1 setup/check` |
| 928 | `GreenFunc1 restore` |
| 929 | `CalH1 threaded combine` |
| 935 | `CalH1 direct proj ratio` |
| 936 | `CalH1 PfM2/IP fused` |

注意:

- 920 以降の細粒度 timer は hot path に overhead を入れる。
- 通常の絶対時間比較には `zvo_CalcTimer.dat` を使い、診断比率には
  `MVMC_CALHAM1_DIAG=1` の `zvo_CalcTimerDiag.dat` を使う。

### 4. Direct projection ratio

対象:

- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl`

内容:

- Hubbard input は Gutzwiller + Jastrow のみで、DH2/DH4/RBM がない。
- `GreenFunc1_real` の hop ごとに `proj_cnt_new` を丸ごと copy/update し、
  `ProjRatio` で全 projection を走査する必要がない。
- `calh1_proj_ratio_no_dh_after_move` を追加し、hop 後の
  Gutzwiller/Jastrow ratio を差分式で直接計算した。
- DH2/DH4/RBM または不完全な index table がある場合は direct path を使わない。

結果:

- `UpdateProjCnt [921]` と `ProjRatio [922]` は CalH1 fast path では 0 になった。
- `CalHamiltonian1 [71]` は `L16/L24/L32` で大きく短縮した。

### 5. PfM2/IP fused path

対象:

- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl`

内容:

- `calculate_new_pf_m2_ip_real` を追加した。
- 従来は:
  1. `calculate_new_pf_m2_real!` が `pf_m_new_real[qp]` を作る。
  2. `calculate_ip_real` が `qp_full_weight[qp] * pf_m_new_real[qp]` を走査する。
- fused path では同じ qp loop 内で `-ratio * pf_m_real[qp]` と weighted sum を計算し、
  `pf_m_new_real` への store と再走査を避けた。

結果:

- `CalculateNewPfM2_real [923]` と `CalculateIP_real [924]` は CalH1 fast path では 0 になった。
- 代わりに `CalH1 PfM2/IP fused [936]` に集約された。
- 効果は direct projection ほど大きくないが、`CalHamiltonian1 [71]` はさらに少し下がった。

### 6. Projection value cache

対象:

- `Julia-mVMC/MVMCOptimizers.jl/src/threading.jl`
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl`

内容:

- `VMCMainCalScratch` に以下を追加した。
  - `calh1_gutz_site_value`
  - `calh1_jastrow_pair_value`
  - projection cache metadata
- `calh1_jastrow_pair_value[r*n_site + c + 1]` に実数 Jastrow 値を展開し、
  hop ごとの pair index lookup と term lookup を避けた。
- `calh1_gutz_site_value[r+1]` に site ごとの実数 Gutzwiller 値を展開した。

結果:

- `CalH1 direct proj ratio [935]` はさらに短縮した。
- 最終版 `projcache-fused` が今回の最良版。

## Diagnostic Breakdown

`zvo_CalcTimerDiag.dat`, warm median seconds。

| L | series | GreenFunc1[920] | Update[921] | ProjRatio[922] | PfM2[923] | IP[924] | Prep[925] | Loop[926] | Setup[927] | Restore[928] | DirectProj[935] | FusedPfIP[936] |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | outer | 0.13663 | 0.01778 | 0.01250 | 0.02203 | 0.00886 | 0.00004 | 0.15416 | 0.01873 | 0.00432 | NA | NA |
| 16 | direct | 0.11761 | 0.00000 | 0.00000 | 0.02169 | 0.00805 | 0.00004 | 0.13495 | 0.01852 | 0.00430 | 0.01705 | NA |
| 16 | fused | 0.11168 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00003 | 0.12916 | 0.01871 | 0.00428 | 0.01782 | 0.02593 |
| 16 | projcache | 0.10417 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00475 | 0.12138 | 0.01902 | 0.00418 | 0.01103 | 0.02526 |
| 24 | outer | 0.22249 | 0.03254 | 0.02135 | 0.04299 | 0.01285 | 0.00004 | 0.24880 | 0.02799 | 0.00636 | NA | NA |
| 24 | direct | 0.19191 | 0.00000 | 0.00000 | 0.04213 | 0.01247 | 0.00005 | 0.21822 | 0.02809 | 0.00661 | 0.03043 | NA |
| 24 | fused | 0.18161 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00004 | 0.20811 | 0.02908 | 0.00633 | 0.03064 | 0.04852 |
| 24 | projcache | 0.16827 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.01042 | 0.19440 | 0.02792 | 0.00637 | 0.01784 | 0.04814 |
| 32 | outer | 0.33377 | 0.05471 | 0.03346 | 0.07386 | 0.01849 | 0.00004 | 0.36965 | 0.03821 | 0.00869 | NA | NA |
| 32 | direct | 0.28726 | 0.00000 | 0.00000 | 0.07395 | 0.01815 | 0.00003 | 0.32352 | 0.03782 | 0.00856 | 0.05013 | NA |
| 32 | fused | 0.27095 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00004 | 0.30739 | 0.03830 | 0.00854 | 0.05003 | 0.08136 |
| 32 | projcache | 0.24705 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.01578 | 0.28292 | 0.03887 | 0.00837 | 0.02679 | 0.07863 |

Reading:

- Direct projection が最大の改善要因。
- PfM2/IP fused は小さいが一貫して効いた。
- Projection value cache は `DirectProj [935]` をほぼ半減した。
- `Setup [927]` と `Restore [928]` は残るが、絶対値は小さい。

## Artifacts

Final source:

- `/home/pj25000164/ku50001833/shin-mvmc-v04-mpi-benchmark/source/Julia-mVMC-calham1-projcache-fused-20260617-141427`

Final local results:

- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-projcache-fused-normal-L16-R4-20260617-141427`
- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-projcache-fused-normal-L24-R4-20260617-141427`
- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-projcache-fused-normal-L32-R4-20260617-141427`
- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-projcache-fused-diag-L16-R4-20260617-141427`
- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-projcache-fused-diag-L24-R4-20260617-141427`
- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-projcache-fused-diag-L32-R4-20260617-141427`

Important intermediate results:

- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-slater-fastpath-normal-L16-R4-20260617-115522`
- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-slater-fastpath-normal-L24-R4-20260617-115522`
- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-slater-fastpath-normal-L32-R4-20260617-115522`
- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-fastpath-normal-L16-R4-20260617-132928`
- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-directproj-normal-L16-R4-20260617-135858`
- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-directproj-fused-normal-L16-R4-20260617-140434`

C comparison:

- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-perf-ws123-timing-c-v14-fixed-input-20260615-211451`

## Verification

Local tests after the final `projcache-fused` implementation:

- `julia --project=. test_unit/test_unit_vmc_main_cal_sr.jl` pass
- `julia --project=. -t 2 test_unit/test_unit_vmc_main_cal_sr.jl` pass
- `julia --project=. test_unit/test_unit_vmc_sampling_proj.jl` pass
- `julia --project=. test_unit/test_unit_threading.jl` pass
- `julia --project=. test/runtests.jl` pass
- `git diff --check` pass

Genkai:

- Final normal jobs completed for `L=16,24,32`, ranks `4`, repeats `3`.
- Final diagnostic jobs completed for `L=16,24,32`, ranks `4`, repeats `3`.
- `zvo_CalcTimer.dat` and `zvo_CalcTimerDiag.dat` were recovered locally.

## Remaining Work

CalHamiltonian1 is now close enough to stop as a first pass. The next useful
performance questions are:

1. `CalculateMAll [40]`: C `icc` and Julia are close but not uniformly ordered.
   Inspect run-to-run variance and possible layout/kernel differences before
   changing code.
2. Overall variance: `All [0]` and `VMCMainCal [4]` fluctuate enough that a
   single median can be misleading. Compare `run_01` / `run_02` / `run_03` and
   section-level variance for Julia and C.
3. Rank imbalance: `WeightAverage` diagnostics from 2026-06-15 suggested that
   collective wait can appear in later sections. If `All [0]` remains noisy,
   collect rank-local section times or per-rank logs rather than only rank0
   `zvo_CalcTimer.dat`.
