# CalHamiltonian1 / ReturnSlaterElmDiff gap on the Hubbard real path (issue #207)

Related to #185, #205 (harness), #207. Follows `hubbard_chain_2026-10-03_sections.md`.

## Summary

* The `VMCMainCal` gap of the 2026-10-03 memo is closed. With identical outputs, Rust
  needs 0.66-0.69x of its previous time on the three report inputs (300 SR steps), and is
  now 1.20-1.26x **faster** than Julia 1.13.1 (it was 0.74x). `|dE|` against Julia is 0.
* Section timers (20 steps, 32 sites): `CalHamiltonian1` 0.330 -> 0.115 s (C 0.137,
  Julia 0.137), `ReturnSlaterElmDiff` 0.398 -> 0.138 s (C 0.177, Julia 0.129),
  `CalculateNewPfM2` 0.166 -> 0.081 s, `UpdateMAll` 0.762 -> 0.534 s, `VMCMainCal`
  2.460 -> 1.651 s. Every changed section is now at or below the native C section time.
* What is left: `CalculateMAll` / `recal PfM and InvM` (Pfaffian setup, not in the scope
  of #207) is 1.2x the native C time at 32 sites (1.352 vs 1.114 s) although faster than
  Julia (1.612 s); `All` is 1.02x the C total at 32 sites and 0.87x at 16 sites.
* Numerics: every optimization output file (`zvo_out.dat`, `zqp_opt.dat`) is
  **bit-identical** before and after for every cell below, so the C operation order,
  RNG draws and acceptance decisions are untouched; no tolerance was changed.

## Environment

* Linux 6.8 x86_64, Intel Xeon E5-2699 v3 (18 cores/36 threads, 2.3 GHz nominal), shared
  host; host load average (1 min) during the runs is in the tables (0.7-1.6).
* rustc 1.99.0, release profile (`opt-level=3`, thin LTO, `debug=1`); OpenBLAS 0.3.26
  pthread (`/usr/lib/x86_64-linux-gnu/openblas-pthread`); `OPENBLAS_NUM_THREADS=1`,
  `OMP_NUM_THREADS=1`; 1 MPI rank, 1 thread everywhere, `MVMC_RS_INNER_THREADS` unset.
* Julia 1.13.1 with the checked-in `Manifest-v1.13.toml`, same BLAS, `JULIA_NUM_THREADS=1`.
* C: `extern/mVMC-1.3.0` `vmc.out` built `-O3` (gcc 13.3) in the repository Dev Container
  image (MPICH, `mpiexec -n 1`, container OpenBLAS), same inputs with only
  `NSROptItrStep`/`NSROptItrSmp` set to the step count. C runs in Docker on the same
  host; its section timers are the C-native `zvo_CalcTimer.dat`.
* "before" is `origin/main` `8dc757d4` (PR #412), "after" is this change; the release
  binaries were built separately and run **interleaved** (`rep -> L -> before, after`).
* Inputs: `benchmark/hubbard_chain/inputs/hubbard_chain_L{16,24,32}`.

## Rust before/after (300 SR steps, internal run time)

Median (min-max) seconds of the `Completed N SR steps in X s` line, 5 interleaved reps.
Raw rows: `hubbard_chain_2026-10-06_calh1_slaterdiff.csv`.

| sites | reps | before median (min-max) s | after median (min-max) s | after/before | host load1 | output digests (zvo_out, zqp_opt) |
|---:|---:|---:|---:|---:|---|---|
| 16 | 5 | 12.96 (12.94-13.02) | 8.60 (8.56-8.64) | 0.66 | 0.95-1.62 | identical |
| 24 | 5 | 27.88 (27.73-27.92) | 18.77 (18.71-18.79) | 0.67 | 0.97-1.34 | identical |
| 32 | 5 | 50.73 (50.23-51.00) | 34.81 (34.78-35.15) | 0.69 | 1.00-1.19 | identical |

## Rust vs Julia (`xtask bench-hubbard`, 300 steps, 3 reps, 1 warm-up, 1 thread)

| model | Rust median (s) | Julia median (s) | speedup (julia/rust) | \|dE\| |
|---|---:|---:|---:|---:|
| hubbard_chain_L16 | 8.630 | 10.853 | 1.258x | 0.00e0 |
| hubbard_chain_L24 | 18.710 | 23.367 | 1.249x | 0.00e0 |
| hubbard_chain_L32 | 34.960 | 41.786 | 1.195x | 0.00e0 |

(`speedup > 1` means Rust is faster. Host load at report time 1.0.) Before the change the
same command gave 0.74x/0.72x/0.74x on the Mac reference
(`benchmark/hubbard_chain/README.md`) and 0.84x/0.82x at 16/32 sites (20 steps) on this
Linux host in `hubbard_chain_2026-10-06_inner_threads.md`.

## Section timers (`MVMC_C_TIMER=1`, 20 SR steps, median of 3, seconds)

Rust before/after and Julia/C runs alternate by repetition on the same host (load
0.7-1.2). Timers are inclusive.

### 16 sites

| section | Rust before | Rust after | C (native) | Julia 1.13.1 | after/before | after/C |
|---|---:|---:|---:|---:|---:|---:|
| `All [0]` | 1.061 | 0.744 | 0.852 | 0.886 | 0.70x | 0.87x |
| `VMCMakeSample [3]` | 0.394 | 0.304 | 0.393 | 0.312 | 0.77x | 0.77x |
| `hopping update [32]` | 0.251 | 0.168 | 0.268 | 0.147 | 0.67x | 0.63x |
| `UpdateMAll [63]` | 0.159 | 0.095 | 0.138 | 0.059 | 0.60x | 0.69x |
| `CalculateNewPfM2 [61]` | 0.044 | 0.025 | 0.087 | 0.021 | 0.57x | 0.28x |
| `recal PfM and InvM [34]` | 0.124 | 0.117 | 0.109 | 0.139 | 0.94x | 1.07x |
| `VMCMainCal [4]` | 0.649 | 0.422 | 0.438 | 0.509 | 0.65x | 0.96x |
| `CalculateMAll [40]` | 0.345 | 0.324 | 0.307 | 0.395 | 0.94x | 1.06x |
| `LocEnergyCal [41]` | 0.104 | 0.038 | 0.053 | 0.047 | 0.37x | 0.72x |
| `CalHamiltonian1 [71]` | 0.102 | 0.036 | 0.041 | 0.045 | 0.36x | 0.87x |
| `ReturnSlaterElmDiff [42]` | 0.111 | 0.041 | 0.065 | 0.042 | 0.37x | 0.63x |

### 32 sites

| section | Rust before | Rust after | C (native) | Julia 1.13.1 | after/before | after/C |
|---|---:|---:|---:|---:|---:|---:|
| `All [0]` | 4.112 | 2.971 | 2.916 | 3.327 | 0.72x | 1.02x |
| `VMCMakeSample [3]` | 1.611 | 1.279 | 1.415 | 1.219 | 0.79x | 0.90x |
| `hopping update [32]` | 1.068 | 0.755 | 1.003 | 0.618 | 0.71x | 0.75x |
| `UpdateMAll [63]` | 0.762 | 0.534 | 0.696 | 0.330 | 0.70x | 0.77x |
| `CalculateNewPfM2 [61]` | 0.166 | 0.081 | 0.204 | 0.078 | 0.49x | 0.40x |
| `recal PfM and InvM [34]` | 0.501 | 0.481 | 0.378 | 0.551 | 0.96x | 1.27x |
| `VMCMainCal [4]` | 2.460 | 1.651 | 1.464 | 1.936 | 0.67x | 1.13x |
| `CalculateMAll [40]` | 1.412 | 1.352 | 1.114 | 1.612 | 0.96x | 1.21x |
| `LocEnergyCal [41]` | 0.333 | 0.118 | 0.149 | 0.140 | 0.35x | 0.79x |
| `CalHamiltonian1 [71]` | 0.330 | 0.115 | 0.137 | 0.137 | 0.35x | 0.84x |
| `ReturnSlaterElmDiff [42]` | 0.398 | 0.138 | 0.177 | 0.129 | 0.35x | 0.78x |

## What was slow and what changed

Self-attribution before (32 sites, 20 steps, `MVMC_*_DIAG`; diagnostic timers add
overhead, so ratios are the signal): CalH1 `GreenFunc1Real [920]` 0.423 s
(`CalculateNewPfM2_real` 0.173, `ProjRatio` 0.089, `UpdateProjCnt` 0.047,
`GreenFunc1 setup/check` 0.034, `CalculateIP_real` 0.011); `Slater buffer accumulate
[933]` 0.271 s and `Slater transOrb build [932]` 0.102 s; `VMCMainCal unmeasured [940]`
0.291 s of which `post CalculateMAll [943]` 0.280 s (copying the real inverse into the
complex tables every sample).

* **CalH1 transfer fast path** (`calh1_real_fast`, diagnostics 925/926/929/935/936 now
  nonzero): serial real transfer section without the per-term scratch copies;
  the direct projection ratio reads flattened Gutzwiller/Jastrow tables built once per
  parameter signature and the unmoved occupations (only the touched source/destination
  entries are adjusted), skipping `UpdateProjCnt`; `calculate_new_pf_m2_ip_real_flat`
  fuses `PfM2` and `IP`. The sums (`z`, the 32-term inner sum, the QP-ordered IP and the
  term-ordered transfer energy) keep the C/Julia operation order. Lanczos and the
  parallel (`MVMC_RS_INNER_THREADS > 1`, above the #361 gates) transfer paths keep the
  generic kernel.
* **Slater derivative**: the real main calculation accumulates the QP/orbital buffer on
  the real inverse and Pfaffian tables (the complex path multiplied zero imaginary parts;
  results are exactly the same real parts), uses the cached orbital matrices instead of
  rebuilding them every sample, and builds the translation tables once per sector row
  instead of per matrix entry. Because nothing else reads them, the complex inverse is no
  longer refreshed after `CalculateMAll` for real optimization runs (still done for
  PhysCal/Lanczos, FSZ and models whose QP weights have imaginary parts).
* **Sampling kernels**: `CalculateNewPfM2` advances four QPs together to overlap the
  latency of each QP's serial sum chain; the rank-one inverse update and the inverse
  assembly use slice loops instead of indexed element access. Each element is produced
  by the same sequence of operations.
* Tests (same-implementation exact checks, not Rust-vs-C float comparisons):
  `real_fast_path_matches_complex_path_on_real_data`,
  `direct_projection_tables_match_indexed_lookups_bitwise` (also the moved-occupation
  ratio), `slice_based_rank_one_update_matches_indexed_reference_bitwise`,
  `blocked_new_pf_m2_matches_per_qp_reference_for_every_qp_count`,
  `fused_new_pf_m2_ip_matches_separate_kernels_bitwise`; the timer-presence callback test
  now expects 925/926/929/935/936 on the Hubbard real path. The existing
  C-/Julia-fixture, SR/CG, PhysCal and Lanczos suites pass unchanged.

## Limitations

* Single host (shared, load 0.7-1.6) and one binary pair; the 300-step cells have
  min-max ranges under 2%, the 20-step section tables are medians of 3.
* The C numbers come from a container build (gcc 13.3 `-O3`, container OpenBLAS) on the
  same CPU; they are a reference for section ratios, not a statement about C's best
  build.
* Real-mode Hubbard-chain inputs only. The CalH1 fast path needs the model without RBM,
  doublon-holon projection or inconsistent Gutzwiller/Jastrow indices (the same
  eligibility as the previous direct projection ratio); other models take the generic
  kernels as before.
* `CalculateMAll` (Pfaffian setup) remains 1.06-1.21x the C section time; it is outside
  this issue.
