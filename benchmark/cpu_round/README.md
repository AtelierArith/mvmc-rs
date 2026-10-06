# CPU performance round (issue #448)

Profile of the major modes against native C and Julia, hotspot ranking, and before/after of the fixes.
Host: Intel Xeon E5-2699 v3 (36 threads), Linux 6.8 x86_64, glibc 2.39. Rust: release profile. C: unpatched
mVMC 1.3.0 `vmc.out` built by `c_toolbox/perf_448/build_c.sh` (CMake Release, MPICH, Dev Container, one rank).
Julia 1.13.1 (`extern/Julia-mVMC`, `Manifest-v1.13.toml`), one thread. The host is shared with other agents, so
every CSV row carries the load average before/after and `*.meta` the load at start/end; medians of 3 (2 for the
thread sweep), Rust and C interleaved per workload. Machine-readable data is in `results/`.

Harnesses: `scripts/bench_cpu_round.py` (Rust vs C, section timers from `output/zvo_CalcTimer.dat`, which both
programs write in the same format), `scripts/bench_cpu_ab.py` (two Rust binaries, also checks that all numerical
output files are byte-identical), `cargo run -p xtask -- bench-hubbard` (Julia). perf was run through
`c_toolbox/perf_448` style privileged container (`perf_event_paranoid=4` on the host), `-e cpu-clock`,
`--call-graph dwarf`.

## Results, one thread (seconds, median)

| workload | Rust before | Rust after | C | C / Rust after |
|---|---|---|---|---|
| opt_hubbard_L16 | 1.22 | 1.21 | 1.68 | 1.40 |
| opt_hubbard_L32 | 3.92 | 3.69 | 4.44 | 1.20 |
| opt_hubbard_L64 | 6.54 | 6.39 | 6.94 | 1.09 |
| opt_heisenberg_real | 6.66 | 6.55 | 8.42 | 1.29 |
| opt_heisenberg_cmp | 14.24 | 13.66 | 18.07 | 1.32 |
| opt_heisenberg_fsz | 13.27 (*) | 4.12 | 4.67 | 1.13 |
| opt_kondo_real | 9.29 | 7.49 | 9.71 | 1.30 |
| opt_hubbard_dh | 5.45 | 4.44 | 11.63 | 2.62 |
| opt_hubbard_rbm_opttrans | 12.48 | 8.28 | 11.40 | 1.38 |
| phys_hubbard_L16 | 2.92 | 2.75 | 4.07 | 1.48 |
| phys_hubbard_L32 | 10.42 | 10.44 | 16.11 | 1.54 |
| phys_heisenberg_real | 14.26 | 14.21 | 18.07 | 1.27 |
| phys_lanczos | 86.18 | 58.85 | 36.31 | 0.62 |

(*) the baseline sweep ran under heavy host load for this workload; the quiet interleaved A/B of the two binaries
(`ab_rust_before_after.csv`) gives 5.23 s -> 4.09 s (1.28x). Differences of a few percent on the other rows are
within load noise; the A/B file is the reliable before/after for the code changes.

Byte-identity: `bench_cpu_ab.py` compared every `zvo_*`/`zqp_*` output of the before/after binaries on
opt_heisenberg_fsz, opt_hubbard_rbm_opttrans, opt_heisenberg_cmp, opt_hubbard_dh, opt_kondo_real,
opt_hubbard_L32 and phys_lanczos: all identical.

## Hotspots found and fixed

1. **Complex stored-O Gram (`multiply store OO`, FSZ and RBM/complex optimization)**: the C-order per-entry sum
   (`sr_backend::c_order_gram_complex`) streamed the whole store once per output entry, 6x to 16x slower than C's
   ZGEMM (section [45]). Now a sample-outer, four-column-block kernel on plain `f64` lanes (no FMA, per-entry
   sample order unchanged, so bit-identical; unit test against the plain loop plus a timing aid). Gram micro
   benchmark 157 ms -> 30 ms. FSZ opt 1.28x, RBM opttrans 1.40x faster.
2. **Lanczos PhysCal**: every Hamiltonian term cloned the complete `SlaterMatrixData` (including the Slater
   elements) to restore it; now only the tables that change are copied back in place
   (`SlaterMatrixData::restore_tables_from`). 86 s -> 59 s (1.46x), byte-identical.

## Hotspots ticketed

- #478 Lanczos PhysCal still 1.6x slower than C: C applies a rank-one `UpdateMAll` per moved configuration, Rust
  recomputes the full Pfaffian/inverse per term (`calc_m_all_*`, 59 % of the time). Needs a faithful port of
  `calHCA1/calHCACA1/checkGF` and changes the operation sequence, so it is separate work.
- #479 The inner-thread pool does not scale where C's OpenMP does (L64 optimization, L32 PhysCal at 4 to 16
  threads; see `results/inner_*.csv`). Forcing the pool (`MVMC_RS_INNER_THRESHOLD=1`) is 1.3x to 14x slower, so the
  default work gate is correct; the per-region dispatch cost is the limit.

## Not hotspots

Single thread Rust is faster than C on every other measured mode (1.09x to 2.6x). Remaining section-level
differences (Heisenberg `CalHamiltonian2` 1.16x, `ReturnSlaterElmDiff` on the RBM input 1.35x) are small
fractions of the run.

## Julia (Rust vs Julia 1.13.1, 30 steps, one thread, `results/julia_hubbard_chain.csv`)

| input | Rust | Julia | Julia / Rust | abs dE |
|---|---|---|---|---|
| hubbard_chain_L16 | 0.850 | 1.242 | 1.46 | 1.1e-16 |
| hubbard_chain_L24 | 1.910 | 2.741 | 1.44 | 2.2e-16 |
| hubbard_chain_L32 | 3.570 | 4.984 | 1.40 | 3.8e-15 |
| hubbard_chain_L64 | 18.050 | 22.992 | 1.27 | 3.8e-15 |

## Thread sweep (`MVMC_RS_INNER_THREADS`, `results/inner_*.csv`, host load 1 to 8)

| workload | Rust 1T | 4T | 8T | 16T | 8T forced pooling | C 1T | C 4T | C 8T |
|---|---|---|---|---|---|---|---|---|
| opt_hubbard_L64 | 6.37 | 6.36 | 6.33 | 6.37 | 8.39 | 7.04 | 2.90 | 2.33 |
| phys_hubbard_L32 | 10.26 | 10.39 | 10.32 | 10.30 | 23.11 | 16.05 | 7.99 | 7.40 |
| opt_heisenberg_cmp | 13.63 | 13.80 | 13.71 | 13.65 | 51.72 | 18.12 | 14.68 | 17.67 |
| opt_heisenberg_fsz | 4.08 | 4.10 | 3.91 | 3.81 | 59.05 | 4.65 | 8.28 | 11.21 |
| opt_hubbard_rbm_opttrans | 8.24 | 8.05 | 7.52 | 7.30 | 59.99 | 11.32 | 10.56 | 13.11 |

(The C 8T column was measured under load 2 to 5, so treat it as indicative; the C 4T/8T values on the small models
are slower than 1T, which is C's OpenMP overhead.)
