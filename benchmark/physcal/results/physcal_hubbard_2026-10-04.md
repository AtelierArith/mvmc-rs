# Hubbard-chain PhysCal benchmark (2026-10-04)

Fixed-parameter (`NVMCCalMode=1`) PhysCal at the `bench-hubbard` Hubbard-chain
sizes, one thread. Warmup-excluded medians over 3 repetitions. Rust is
`mvmc-cli --physcal` process wall clock; Julia is
`run_phys_cal_from_namelist` wall clock with JIT excluded by the warm-up run
(Darwin arm64, rustc 1.99.0, Julia 1.13.1).

| model | Rust median (s) | Julia median (s) | speedup (julia/rust) | \|ΔE\| |
|---|---:|---:|---:|---:|
| hubbard_chain_L16 | 1.910 | 1.070 | 0.560x | 0.00e0 |
| hubbard_chain_L24 | 4.070 | 2.174 | 0.534x | 0.00e0 |
| hubbard_chain_L32 | 7.920 | 4.074 | 0.514x | 0.00e0 |

Rust is ~1.8x slower across sizes. The committed inputs use `NDataQtySmp=100`,
so startup is negligible and the gap is per-sample work; the section breakdown
is in `physcal_hubbard_2026-10-04_sections.md` and points at `VMCMainCal`
(`CalHamiltonian1`), the same kernel location as issue #207 for the SR path.

Command:

```sh
cargo run -p xtask -- bench-physcal-hubbard --reps 3 --warmups 1 --threads 1
```
