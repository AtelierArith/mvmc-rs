---
date: 2026-06-17
datetime: 2026-06-17 17:55 JST
model: GPT-5 Codex
status: report
topic: Julia-mVMC Hubbard optimization final summary
related:
  - docs/reports/2026-06-17-julia-mvmc-hubbard-locenergy-slater-calham1-optimization-record.md
  - docs/reports/2026-06-17-julia-mvmc-hubbard-calculatemall-variance-check.md
  - docs/reports/2026-06-17-julia-mvmc-v041-ohtaka-genkai-runtime-comparison.md
  - docs/reports/2026-06-15-julia-mvmc-hubbard-ws123-performance-diagnostics.md
---

# Julia-mVMC Hubbard optimization final summary

## Conclusion

Genkai Hubbard real-mode, `L=16,24,32`, `R=4` の速度比較と kernel
最適化は、いったん完了とする。

最終判断:

- `ReturnSlaterElmDiff [42]` は C `icc` とほぼ同等。
- `CalHamiltonian1 [71]` は C `gcc` より速く、C `icc` との差は
  `7%` から `14%` 程度まで縮小。
- `LocEnergyCal [41]` は C `icc` 比で `1.02x` から `1.08x` 程度。
- `CalculateMAll [40]` は C `icc` とほぼ同等か少し速い。C `gcc` はなお強いが、
  現時点の Julia 全体差の主因ではない。
- `VMCMainCal [4]` の run-to-run spike は、local accumulator / scratch の
  毎 step 再確保が主因で、workspace reuse により解消。
- `WeightAverage [21]` の spike は WeightAverage 自身の計算・通信量ではなく、
  rank 間の `VMCMakeSample [3]` / `VMCMainCal [4]` 到達差が最初の allreduce で
  可視化されたもの。

したがって、これ以上の速度比較は深追いしない。次に進むなら benchmark ではなく、
コード整理、回帰テスト、変更単位の分割、または `VMCMakeSample [3]` の
load-balance 調査である。

## Implemented Code Changes

主な変更:

- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl`
  - Hubbard real-mode の `CalHamiltonian1 [71]` fast path。
  - direct projection ratio path。
  - `calculate_new_pf_m2_real!` / `CalculateIP_real` の fused path。
  - Slater fast path の scratch buffer / transOrb build / buffer accumulate 改善。
  - `MVMC_CALHAM1_DIAG`, `MVMC_SLATER_DIAG`, `MVMC_MAINCAL_DIAG` の gated diagnostic。
- `Julia-mVMC/MVMCOptimizers.jl/src/threading.jl`
  - `VMCMainCalScratch` の sample-local buffer、Slater buffer、CalH1 cache。
  - `VMCThreadAccumulator` / `VMCSROptAccumulator` / `VMCMainCalScratch` の
    `SamplingWorkspace` reuse。
  - SR store mode 向けの mode-aware reset。
- `Julia-mVMC/MVMCOptimizers.jl/src/types.jl`
  - `SamplingWorkspace.main_cal_accumulator` を追加。
- `Julia-mVMC/MVMCOptimizers.jl/src/weight_average.jl`
  - `MVMC_WEIGHTAVG_DIAG` 用の WeightAverage sub-timer。
- `Julia-mVMC/MVMCOptimizers.jl/src/c_timer.jl`
  - diagnostic timer `920` から `966`。
  - diagnostic env helper。
- `Julia-mVMC/MVMCOptimizers.jl/src/run_para_opt_from_namelist.jl`
  - diagnostic env による timer enable と `zvo_CalcTimerDiag.dat` 出力。
  - `MVMC_WEIGHTAVG_DIAG=1` の場合は rank-local
    `zvo_rankXXXX_CalcTimer*.dat` も出力。

## Final Measurements

重要な実測:

| item | before | after |
|---|---:|---:|
| `diag[941]` accumulator init/reset median | 0.01432s | 0.00260s |
| `diag[941]` accumulator init/reset max | 0.22726s | 0.00420s |
| `VMCMainCal[4]` max, diagnostic run | 1.32371s | 1.11354s |
| `VMCMainCal[4]` max, normal run | 1.28246s | 1.10863s |
| normal unaccounted max | 0.21856s | 0.04199s |
| rank0 `WeightAverage[21]` max in weightavg diagnostic | 0.76176s | wait in `WE allreduce[961]` |
| rank0 `SR[25]` median in weightavg diagnostic | 0.00239s | stable |

Key run directories:

- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-projcache-fused-normal-L16-R4-20260617-141427`
- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-projcache-fused-normal-L24-R4-20260617-141427`
- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-calham1-projcache-fused-normal-L32-R4-20260617-141427`
- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-maincalws-normal-L24-R4-repeat10-20260617-171858`
- `benchmark/heisenberg_chain_v04_mpi_genkai/results/julia-hubbard-weightavgdiag-L24-R4-repeat10-20260617-173941`

## Diagnostic Flags

Diagnostic flags are off by default.

| env | purpose |
|---|---|
| `MVMC_C_TIMER=1` | C-compatible `zvo_CalcTimer.dat` |
| `MVMC_CALHAM1_DIAG=1` | CalH1 / GreenFunc1 real diagnostic, ids `920-929` |
| `MVMC_SLATER_DIAG=1` | Slater diagnostic, ids `930-936` |
| `MVMC_MAINCAL_DIAG=1` | VMCMainCal unmeasured diagnostic, ids `940-950` |
| `MVMC_WEIGHTAVG_DIAG=1` | WeightAverage rank-local diagnostic, ids `960-966` |

## Remaining Work

Performance work that is still meaningful, but not required for the current
speed-comparison milestone:

- `VMCMakeSample [3]` rank-local load-balance diagnostic.
- `CalculateMAll [40]` vs C `gcc` gap, if C `gcc` parity becomes the target.
- Longer clean repeat on final code without diagnostic flags, if release notes need
  a single final timing table.

Current priority should be code review / regression test / change splitting rather
than additional benchmark exploration.
