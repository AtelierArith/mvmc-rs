# Hubbard-chain section-timer breakdown (2026-10-03)

Memo for the Rust-vs-Julia gap observed in
`hubbard_chain_2026-10-03.md`. Produced with the C-compatible section timer
(`MVMC_C_TIMER=1`) on the committed inputs: 300 SR steps, one thread, one run
per side. Timers are inclusive, so nested sections overlap.

## L=32 (largest, clearest)

| section | Rust | Julia | ratio | gap |
|---|---:|---:|---:|---:|
| `VMCMainCal [4]` | 11.605s | 8.042s | 1.44x | +3.56s |
| `  LocEnergyCal [41]` | 3.070s | 0.842s | 3.65x | +2.23s |
| `    CalHamiltonian1 [71]` | 3.057s | 0.832s | 3.68x | +2.23s |
| `  ReturnSlaterElmDiff [42]` | 1.939s | 0.642s | 3.02x | +1.30s |
| `VMCMakeSample [3]` | 4.560s | 3.795s | 1.20x | +0.76s |
| `  hopping update [32]` | 3.358s | 2.611s | 1.29x | +0.75s |
| `    UpdateMAll [63]` | 1.846s | 1.227s | 1.50x | +0.62s |
| `    CalculateNewPfM2 [61]` | 0.720s | 0.407s | 1.77x | +0.31s |
| `outputData [22]` | 0.528s | 0.108s | 4.89x | +0.42s |
| `All [0]` | 16.762s | 12.412s | 1.35x | +4.35s |

## L=16 (same ordering)

| section | Rust | Julia | ratio |
|---|---:|---:|---:|
| `All [0]` | 4.453s | 3.136s | 1.42x |
| `VMCMainCal [4]` | 2.899s | 2.034s | 1.43x |
| `  LocEnergyCal [41]` | 0.756s | 0.271s | 2.79x |
| `    CalHamiltonian1 [71]` | 0.744s | 0.263s | 2.83x |
| `  ReturnSlaterElmDiff [42]` | 0.558s | 0.201s | 2.78x |
| `VMCMakeSample [3]` | 1.247s | 0.928s | 1.34x |
| `  hopping update [32]` | 0.909s | 0.603s | 1.51x |
| `    UpdateMAll [63]` | 0.435s | 0.232s | 1.87x |
| `    CalculateNewPfM2 [61]` | 0.170s | 0.095s | 1.80x |
| `outputData [22]` | 0.286s | 0.055s | 5.22x |

## Rust `CalHamiltonian1` self-attribution (L=16 diagnostics)

`zvo_CalcTimerDiag.dat` with `MVMC_C_TIMER=1 MVMC_CALHAM1_DIAG=1 MVMC_SLATER_DIAG=1`:

| diagnostic | seconds |
|---|---:|
| `CalH1 GreenFunc1Real [920]` | 1.674 |
| `  ProjRatio [922]` | 0.341 |
| `  CalculateNewPfM2_real [923]` | 0.229 |
| `  GreenFunc1 setup/check [927]` | 0.131 |
| `  UpdateProjCnt [921]` | 0.080 |
| `  CalculateIP_real [924]` | 0.041 |
| `  CalH1 fast prep [925]` / `fast term loop [926]` / `threaded combine [929]` | 0.000 |
| `  direct proj ratio [935]` / `PfM2/IP fused [936]` | 0.000 |
| `SlaterElmDiff_fcmp [930]` | 0.562 |
| `  Slater buffer accumulate [933]` | 0.348 |
| `  Slater transOrb build [932]` | 0.145 |

## Interpretation

- The gap is concentrated in the per-sample measurement, not sampling:
  `VMCMainCal` accounts for +3.56s of the +4.35s L=32 gap, and
  `CalHamiltonian1` + `ReturnSlaterElmDiff` account for ~+3.53s of that.
- Rust runs the generic per-transfer-term path
  (`green_func1` -> `UpdateProjCnt` -> `ProjRatio` ->
  `calculate_new_pf_m2_real_flat` -> `calculate_ip_real`). The Julia CalH1
  fast path (diagnostics 925/926/929) and fused PfM2+IP / projcache
  (935/936) are not implemented (all zero).
- `ReturnSlaterElmDiff` is dominated by `Slater buffer accumulate [933]`.
- Sampling (`VMCMakeSample`) is within ~1.2-1.5x and contributes little.

Caveat: single-run, inclusive timers with diagnostic overhead; ratios are the
robust signal. Tracked as issue #207.
