# Hubbard-chain PhysCal section-timer breakdown (2026-10-04)

Memo for the Rust-vs-Julia gap in `physcal_hubbard_2026-10-04.md`. Produced
with the C-compatible section timer (`MVMC_C_TIMER=1`) and the PhysCal input
set under `benchmark/physcal/inputs/`: `NDataQtySmp=100`, one thread,
warm-up-excluded medians over 3 runs per side. Timers are inclusive, so nested
sections overlap.

The Rust timer is the `mvmc-cli --physcal` implementation added for issue
#251. The Julia timer is `run_phys_cal_from_namelist` / `vmc_phys_cal!` on the
`julia-patch` branch (PR tmisawa/Julia-mVMC#54), run via
`--julia-root <julia-patch checkout>`.

## L32 (largest, clearest)

| section | Rust | Julia | ratio R/J |
|---|---:|---:|---:|
| `All [0]` | 9.267 | 5.195 | 1.78 |
| `VMCPhysCal [2]` | 9.266 | 5.191 | 1.79 |
| `  VMCMakeSample [3]` | 1.164 | 1.044 | 1.12 |
| `    hopping update [32]` | 0.888 | 0.753 | 1.18 |
| `      CalculateNewPfM2 [61]` | 0.263 | 0.161 | 1.63 |
| `      UpdateMAll [63]` | 0.327 | 0.225 | 1.45 |
| `  VMCMainCal [4]` | 7.778 | 4.073 | 1.91 |
| `    CalculateMAll [40]` | 2.422 | 2.794 | 0.87 |
| `    LocEnergyCal [41]` | 1.138 | 0.360 | 3.16 |
| `      CalHamiltonian1 [71]` | 1.132 | 0.356 | 3.18 |
| `    CalculateGreenFunc [42]` | 0.705 | 0.000 | n/a |
| `    Lanczos1 [43]` | 0.024 | 0.000 | n/a |
| `  outputData [22]` | 0.326 | 0.059 | 5.54 |

## L16 / L24 (same ordering)

| section | L16 Rust | L16 Julia | R/J | L24 Rust | L24 Julia | R/J |
|---|---:|---:|---:|---:|---:|---:|
| `All [0]` | 1.945 | 1.402 | 1.39 | 4.990 | 2.777 | 1.80 |
| `VMCPhysCal [2]` | 1.945 | 1.401 | 1.39 | 4.989 | 2.774 | 1.80 |
| `  VMCMakeSample [3]` | 0.328 | 0.318 | 1.03 | 0.665 | 0.534 | 1.24 |
| `    hopping update [32]` | 0.241 | 0.213 | 1.13 | 0.506 | 0.377 | 1.34 |
| `      CalculateNewPfM2 [61]` | 0.056 | 0.038 | 1.49 | 0.143 | 0.079 | 1.81 |
| `      UpdateMAll [63]` | 0.085 | 0.058 | 1.46 | 0.159 | 0.096 | 1.66 |
| `  VMCMainCal [4]` | 1.453 | 1.028 | 1.41 | 4.059 | 2.178 | 1.86 |
| `    CalculateMAll [40]` | 0.467 | 0.650 | 0.72 | 1.283 | 1.458 | 0.88 |
| `    LocEnergyCal [41]` | 0.245 | 0.118 | 2.08 | 0.639 | 0.203 | 3.15 |
| `      CalHamiltonian1 [71]` | 0.240 | 0.114 | 2.11 | 0.634 | 0.199 | 3.18 |
| `    CalculateGreenFunc [42]` | 0.174 | 0.000 | n/a | 0.417 | 0.000 | n/a |
| `  outputData [22]` | 0.164 | 0.043 | 3.79 | 0.263 | 0.048 | 5.51 |

## Interpretation

- The gap is in the per-sample measurement, not sampling: `VMCMainCal [4]` is
  ~1.4-1.9x slower while `VMCMakeSample [3]` is only ~1.0-1.2x.
- Inside the measurement, `LocEnergyCal [41]` / `CalHamiltonian1 [71]` is
  2.1-3.2x slower in Rust. This is the same kernel that dominates the SR-path
  gap tracked by issue #207, now reproduced on the PhysCal path.
- `outputData [22]` is 3.8-5.5x slower. Rust writes one indexed file per sample
  (`zvo_out_NNN.dat`, `zvo_var_NNN.dat`, `zvo_cisajs_NNN.dat`, ...), whereas the
  Julia PhysCal runner writes a single `zvo_out.dat` plus per-sample Green
  files; the write shape, not just the bytes, differs.
- `CalculateMAll [40]` is not slower in Rust, so the `VMCMainCal` gap is not a
  blanket matrix-rebuild cost.

## Caveats

- `CalculateGreenFunc [42]` and `Lanczos [43]` are Rust-side instrumentation;
  the Julia `vmc_main_cal!` on the julia-patch branch does not split them under
  these ids. Julia therefore reports 0 and carries the Green work un-timed
  inside `[4]` (about 0.9 s at L32), so the [42]/[43] ratio cells are not
  meaningful and the [4] child sum does not close on the Julia side.
- Section timers add overhead (L32 Rust `[0]` is 9.27 s with the timer vs
  7.96 s without); ratios are the robust signal, not absolute seconds.
- Single machine, one thread, 3 runs per side.

Related to issue #251; the `CalHamiltonian1` location repeats issue #207.
