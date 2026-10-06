# Hubbard-chain PhysCal section-timer breakdown (2026-10-06)

Memo for the Rust-vs-Julia gap in `physcal_hubbard_2026-10-06.md`. Produced on
Linux x86_64 with the release `mvmc` binary and Julia 1.13.1, one thread, system
OpenBLAS on the Rust side and Julia's bundled OpenBLAS, `NDataQtySmp=100`, using
the PhysCal inputs under `benchmark/physcal/inputs/`. Timers are inclusive, so
nested sections overlap.

The Rust timer is the `mvmc-cli --physcal` implementation; the Julia timer is
`run_phys_cal_from_namelist` / `vmc_phys_cal!` from the pinned `extern/Julia-mVMC`
reference. Reproduce with:

```sh
MVMC_C_TIMER=1 cargo run -p xtask -- bench-physcal-hubbard \
  --model hubbard_chain_L32 --reps 1 --warmups 1 --threads 1 --keep-output
```

## L32 sections (warm-up-excluded)

Timer-instrumented run; the same-size clean medians in
`physcal_hubbard_2026-10-06.md` are 9.350s (Rust) and 5.073s (Julia). Timer
overhead inflates both sides, so the ratios are the robust signal.

| section | Rust | Julia | ratio R/J |
|---|---:|---:|---:|
| `All [0]` | 9.543 | 5.264 | 1.81 |
| `VMCPhysCal [2]` | 9.542 | 5.260 | 1.81 |
| `  VMCMakeSample [3]` | 0.919 | 1.086 | 0.85 |
| `    hopping update [32]` | 0.666 | 0.766 | 0.87 |
| `      UpdateMAll [63]` | 0.204 | 0.134 | 1.53 |
| `      CalculateNewPfM2 [61]` | 0.146 | 0.143 | 1.02 |
| `  VMCMainCal [4]` | 8.582 | 4.120 | 2.08 |
| `    CalculateMAll [40]` | 2.007 | 2.770 | 0.72 |
| `    LocEnergyCal [41]` | 0.301 | 0.322 | 0.93 |
| `      CalHamiltonian1 [71]` | 0.295 | 0.316 | 0.93 |
| `    CalculateGreenFunc [42]` | 0.242 | 0.000 | (Julia un-timed) |
| `    Lanczos1 [43]` | 0.023 | 0.000 | — |
| `  outputData [22]` | 0.034 | 0.039 | 0.86 |

## Rust `VMCMainCal` diagnostic (`MVMC_MAINCAL_DIAG=1`)

Run separately on a shared, loaded host, so the absolute seconds are larger than
the table above; the in-run attribution is the signal.

| diagnostic | seconds |
|---|---:|
| `VMCMainCal unmeasured [940]` | 8.796 |
| `  energy accumulate/check [946]` | **7.688** |
| `  post CalculateMAll [943]` | 1.074 |
| `  accumulator init/reset [941]` | 0.005 |
| `  sample copy/check [942]` | 0.008 |
| other `[944]`-`[950]` | < 0.01 |

## Interpretation

- `CalHamiltonian1` is now at parity (0.295 vs 0.316, 0.93x) after #207, and
  `outputData` is at parity (0.034 vs 0.039) after #337. `CalculateMAll` is
  faster in Rust than Julia (0.72x).
- The remaining gap is inside `VMCMainCal`: the Rust child sections sum to
  2.573s of 8.582s, leaving ~6.0s unmeasured. Julia leaves ~1.0s (its Green
  measurement is not split under the same ids).
- `[946] energy accumulate/check` is 7.688s of the 8.796s `[940]` remainder. It
  wraps `calculate_sz`, the weighted energy accumulation and the one/two-body
  Green measurement `ordinary_green_values`; with `NLanczosMode=0` the Lanczos
  `H^2` branch is skipped. The cost is therefore the Green measurement, which
  runs ~6x slower than Julia's un-timed equivalent.
- `ordinary_green_values` calls `green_func1`/`green_func2` once per declared
  term (64 one-body + 192 two-body terms per sample at L32). `green_func2_impl`
  allocates several `Vec`s (`my_ele_idx`, `my_ele_num`, `proj_mid`,
  `proj_final`, `pf_m_new_real`) plus a fresh `GreenScratch`/`CTimer` per call,
  on top of the two-electron Pfaffian update. The per-term allocation is the
  first suspect.

## Caveats

- The `[946]` diagnostic was collected on a loaded host in a separate run; the
  clean section table and the diagnostic table were not measured under identical
  load. Ratios, not absolute seconds, are the robust signal.
- `CalculateGreenFunc [42]` and `Lanczos [43]` are Rust-side instrumentation;
  the Julia runner does not split them under these ids, so its `[42]`/`[43]`
  cells are 0 and carry the Green work inside `[4]`.
- One machine, one thread, one measured run per side for the section table.

Related to issue #442; the `outputData` fix is #337 and the `CalHamiltonian1`
fix is #207.
