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
- #479 (fixed, see the thread sweep below) The inner-thread pool did not scale where C's OpenMP does.

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

## Thread sweep before the spin pool (`MVMC_RS_INNER_THREADS`, `results/inner_*.csv`, host load 1 to 8)

| workload | Rust 1T | 4T | 8T | 16T | 8T forced pooling | C 1T | C 4T | C 8T |
|---|---|---|---|---|---|---|---|---|
| opt_hubbard_L64 | 6.37 | 6.36 | 6.33 | 6.37 | 8.39 | 7.04 | 2.90 | 2.33 |
| phys_hubbard_L32 | 10.26 | 10.39 | 10.32 | 10.30 | 23.11 | 16.05 | 7.99 | 7.40 |
| opt_heisenberg_cmp | 13.63 | 13.80 | 13.71 | 13.65 | 51.72 | 18.12 | 14.68 | 17.67 |
| opt_heisenberg_fsz | 4.08 | 4.10 | 3.91 | 3.81 | 59.05 | 4.65 | 8.28 | 11.21 |
| opt_hubbard_rbm_opttrans | 8.24 | 8.05 | 7.52 | 7.30 | 59.99 | 11.32 | 10.56 | 13.11 |

(The C 8T column was measured under load 2 to 5, so treat it as indicative; the C 4T/8T values on the small models
are slower than 1T, which is C's OpenMP overhead.)

## Lanczos rank-one update (issue #478)

`calculate_lanczos_h2_transfer`, `calculate_lanczos_green` and the exchange/pair-hop terms now move the
Slater tables to the moved configuration with C's `UpdateMAll` (one electron) / `UpdateMAllTwo` (two electrons,
`ml` first, C's `rsb_old` conventions) instead of a full Pfaffian and inverse per term
(`lanczos_move_tables`), and restore them in place (C's `copyMAll`). Configurations that are not a one- or
two-electron move fall back to the full recomputation.

| `phys_lanczos` (1 thread) | seconds (median) | host load |
|---|---|---|
| Rust after #448 (before) | 64.6 (A/B, `lanczos478_ab_rust_before_after.csv`) | 6 |
| Rust #478 (after) | 34.7 (same A/B, 1.86x) | 6 |
| Rust #478 vs native C, interleaved (`lanczos478_vs_c.csv`) | 32.6 vs C 38.5 (1.18x faster) | 2 to 3 |

Deviation from native C (`lanczos478_deviation_from_c.txt`, `scripts/lanczos_deviation_vs_c.py`, largest
absolute difference over the `zvo_ls_*` files relative to the largest magnitude in each file): 3.7e-13 before
and after (the maximum comes from a file whose C/Rust difference is identical for both binaries); the
individual files stay at the 1e-16 to 1e-13 level, so the native-C Lanczos fixtures pass with unchanged
tolerances. The Lanczos output files are no longer byte-identical to the previous Rust binary (rank-one update
instead of full recomputation, as in C); all other outputs of the A/B run are unchanged.

## Thread sweep with the spin pool (issue #479, `results/inner479_inner{1,4,8,16}.csv`)

Diagnosis (L64 optimization, `MVMC_RS_INNER_PROFILE=1`, 8 workers, gate bypassed): the Rayon pool paid 15 to 30 us
per region (wake-up of sleeping workers, `install`, join) and the per-region tables of the Pfaffian kernels. Pooled
`UpdateMAll` (8 planes, 31 us serial) took 26 us and `CalculateNewPfM2` (0.8 us serial) 7.3 us, so only the Pfaffian
region (847 us serial, 300 us pooled) paid off, and the size gate (`n_size >= 138` at 8 workers) rightly kept
everything serial. Fix: a spin pool (`crates/mvmc-core/src/spin_pool.rs`, workers spin for 300 us after each region,
caller runs block 0, static blocks, no per-region allocation: 1 to 2 us per empty 8-block region, measured by the
`dispatch_latency_of_empty_regions` micro-benchmark), the region helpers and the Pfaffian kernels on it with a reusable
staging buffer instead of per-block tables, and gates recalibrated to the new dispatch cost (work estimate of at least
20 us, size gate off). Pooled `UpdateMAll` now takes 6.7 us, `CalculateMAll` at L64 146 us (ideal 110 us); results are
byte-identical for 1, 2, 3, 4 and 8 workers (`threaded_issue182/360/361` invariance tests and the
`ident479` byte comparison of every output file on Hubbard L16/L32, DH, Kondo, RBM+OptTrans, Heisenberg complex and FSZ,
also with the gate bypassed), and the default one-thread outputs are byte-identical to `main` with the same timing
(`results/ab_inner479_1thread_before_after.csv`).

Seconds, median of 2, Rust and native C interleaved (host load before/after every run is in the CSVs; the Rust run
and the 1T/4T C runs saw median load 2.4/2.9, the 8T runs 5.8 and the 16T runs 11.3, so the 8T and 16T columns,
Rust and C alike, are pessimistic):

| workload | Rust 1T | 4T | 8T | 16T | C 1T | C 4T | C 8T | C 16T |
|---|---|---|---|---|---|---|---|---|
| opt_hubbard_L64 | 6.35 | 2.81 | 2.29 | 2.29 | 7.06 | 2.88 | 2.35 | 2.67 |
| opt_hubbard_L32 | 3.69 | 2.61 | 2.62 | 2.63 | 4.45 | 2.55 | 2.45 | 3.34 |
| phys_hubbard_L32 | 10.54 | 7.94 | 6.87 | 6.43 | 16.03 | 8.06 | 7.35 | 8.39 |
| opt_heisenberg_cmp | 13.70 | 13.80 | 13.74 | 13.82 | 18.09 | 14.76 | 17.51 | 23.46 |
| opt_heisenberg_fsz | 4.14 | 4.25 | 3.89 | 3.76 | 4.66 | 8.18 | 11.14 | 16.35 |
| opt_hubbard_rbm_opttrans | 8.22 | 8.13 | 8.68 | 8.82 | 11.33 | 10.49 | 13.15 | 20.63 |

L64 optimization scales 2.8x at 4 workers and 2.8x at 8 (equal to C's OpenMP, 2.35 s), PhysCal L32 1.5x at 8 and 1.6x at
16 (C 7.35 s and 8.4 s), L32 optimization 1.4x. The small models stay at their 1T times (the gate keeps their
sub-20-us regions serial, where C's OpenMP gets slower with threads: 1.9x on FSZ at 8T).

Limits. (1) L32 optimization stops at 1.4x: its `UpdateMAll` regions (8 planes, about 8 us serial) fall below the 20 us
gate. `MVMC_RS_INNER_MIN_WORK_NS=10000` takes L32 to 2.05 s (1.8x) but makes Kondo 35 % slower (7.5 s -> 10.1 s,
its 8-plane `CalculateMAll` costs 11 us serial and about 16 us pooled: the staged kernel carries a fixed
per-region overhead of about 10 us beyond the 1 to 2 us dispatch), so one uniform gate cannot serve both; separate
per-kernel estimates or a cheaper staged Pfaffian region would. (2) The serial remainder (the 0.8 us per-proposal
regions, the local energy at small `n_size`, Slater derivative) bounds the speed-up (Amdahl): at L64 `CalculateMAll`
and `UpdateMAll` were 85 % of the serial time. (3) Timings at 8 and 16 workers were taken while other jobs ran on the
host (load 6 to 11), which lowers every multi-worker number.
