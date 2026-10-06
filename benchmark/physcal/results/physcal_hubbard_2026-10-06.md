# Hubbard-chain PhysCal benchmark (2026-10-06)

Fixed-parameter (`NVMCCalMode=1`) PhysCal at the `bench-hubbard` Hubbard-chain
sizes, one thread. Warm-up-excluded medians over 3 repetitions. Rust is
`mvmc-cli --physcal` process wall clock; Julia is `run_phys_cal_from_namelist`
wall clock with JIT excluded by the warm-up run (Linux x86_64, rustc 1.99.0,
Julia 1.13.1, system OpenBLAS 0.3.26 for Rust and Julia's bundled OpenBLAS).

| model | Rust median (s) | Julia median (s) | speedup (julia/rust) | \|ΔE\| |
|---|---:|---:|---:|---:|
| hubbard_chain_L16 | 1.930 | 1.429 | 0.740x | 0.00e0 |
| hubbard_chain_L24 | 4.660 | 2.841 | 0.610x | 0.00e0 |
| hubbard_chain_L32 | 9.350 | 5.073 | 0.543x | 0.00e0 |

Julia is ~1.35-1.85x faster across sizes. The committed inputs use
`NDataQtySmp=100`, so startup is negligible and the gap is per-sample work. The
section breakdown is in `physcal_hubbard_2026-10-06_sections.md`: `CalHamiltonian1`
(#207) and `outputData` (#337) are now at parity and `CalculateMAll` is faster in
Rust, but the un-timed remainder of `VMCMainCal` — dominated by the one/two-body
Green-function measurement (`energy accumulate/check [946]`, 7.69s of 8.80s at
L32) — is ~6x slower than Julia's equivalent. Tracked by issue #442.

## PhysCal observables (Rust vs Julia)

All 100 per-sample files of each family match, with `max|Δ| = 0` (the
`zvo_cisajscktalt` direct two-body files contain 2 values per row, so L32 has
38400 compared values):

| model | family | files | values | max abs | max rel | status |
|---|---|---:|---:|---:|---:|---|
| hubbard_chain_L16 | zvo_cisajs | 100 | 6400 | 0.000e0 | 0.000e0 | ok |
| hubbard_chain_L16 | zvo_cisajscktalt | 100 | 19200 | 0.000e0 | 0.000e0 | ok |
| hubbard_chain_L24 | zvo_cisajs | 100 | 9600 | 0.000e0 | 0.000e0 | ok |
| hubbard_chain_L24 | zvo_cisajscktalt | 100 | 28800 | 0.000e0 | 0.000e0 | ok |
| hubbard_chain_L32 | zvo_cisajs | 100 | 12800 | 0.000e0 | 0.000e0 | ok |
| hubbard_chain_L32 | zvo_cisajscktalt | 100 | 38400 | 0.000e0 | 0.000e0 | ok |

Command:

```sh
cargo run -p xtask -- bench-physcal-hubbard --reps 3 --warmups 1 --threads 1
```

Related to issue #440 (observable comparison) and #442 (measurement bottleneck).
