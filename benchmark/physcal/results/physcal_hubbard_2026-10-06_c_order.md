# PhysCal C-order kernels, Rust vs Julia (issue #449)

`scripts/run_physcal_benchmark.sh 3 1 1` / `cargo run -p xtask -- bench-physcal-hubbard --reps 3
--warmups 1 --threads 1`. Hubbard chain, 1 thread, `NDataQtySmp = 100`, median of 3 repetitions.
Host: Intel Xeon E5-2699 v3, Linux x86_64 (Ubuntu 24.04), OpenBLAS 0.3.26, Julia 1.13.1, rustc
1.99.0. The host is shared and was loaded by other jobs (load average 4 to 15 during these runs),
so absolute times vary by up to 30 % between runs; the Julia/Rust ratio within one run is the
comparable quantity.

Changes measured: `two_hop_bilinear_real` in C operation order (no FMA, sequential, four-row
interleaving; was Julia's lane-split FMA tree with hardware-FMA dispatch), the complex
`CalculateMAll` of the periodic recalculation and the PhysCal refresh in C order, and the real
Gram through `DGEMM('N','T')` as C (the Gram does not run in PhysCal).

## Interleaved A/B (same host, same session; "old" = sources of `HEAD` before this change)

| model | old run 1 Rust s | old run 2 Rust s | new run 1 Rust s | new run 2 Rust s | Julia/Rust old (1, 2) | Julia/Rust new (1, 2) |
|---|---:|---:|---:|---:|---|---|
| hubbard_chain_L16 | 3.38 | 3.23 | 3.00 | 2.68 | 1.14x, 1.22x | 1.55x, 1.42x |
| hubbard_chain_L24 | 7.09 | 7.12 | 9.96 (load spike) | 5.83 | 1.11x, 1.15x | 0.95x, 1.39x |
| hubbard_chain_L32 | 13.81 | 21.60 (load spike) | 13.57 | 10.79 | 1.29x, 0.75x | 1.03x, 1.32x |

A third, standalone run of the new code (`scripts/run_physcal_benchmark.sh 3 1 1`):
L16 3.22 s vs Julia 4.20 s (1.30x), L24 6.06 s vs 8.41 s (1.39x), L32 10.66 s vs 15.70 s
(1.47x). Reading: the C-order kernel is not slower than the Julia-order one it replaces; the
measured Rust times are 5 to 20 % lower at L16/L24 and Rust stays ahead of Julia at all three
sizes in the quiet runs (the #444 table had 1.12x, 1.06x, 0.98x on a quieter host, so absolute
ratios are not comparable across hosts). The explanation is that the sequential kernel with four
independent row chains needs no software-FMA dispatch and no `% 4` / `% 6` lane arithmetic in the
inner loop. Energies are identical (|dE| = 0 for every run).

## Observables, Rust vs Julia (last repetition of the standalone run)

| model | family | files | values | max abs diff | max rel diff |
|---|---|---:|---:|---:|---:|
| hubbard_chain_L16 | zvo_cisajs | 100 | 6400 | 0 | 0 |
| hubbard_chain_L16 | zvo_cisajscktalt | 100 | 19200 | 2.123e-15 | 1.120e-12 |
| hubbard_chain_L24 | zvo_cisajs | 100 | 9600 | 0 | 0 |
| hubbard_chain_L24 | zvo_cisajscktalt | 100 | 28800 | 3.138e-14 | 8.178e-13 |
| hubbard_chain_L32 | zvo_cisajs | 100 | 12800 | 0 | 0 |
| hubbard_chain_L32 | zvo_cisajscktalt | 100 | 38400 | 2.458e-13 | 5.826e-10 |

Before this change these families were bit-equal to Julia (max|diff| = 0); the roundoff-level
differences are the Julia-to-C reduction-order change (the Julia-order reductions were the only
source of the previous exact equality). Raw timings: `physcal_hubbard_2026-10-06_c_order_runs.csv`.
