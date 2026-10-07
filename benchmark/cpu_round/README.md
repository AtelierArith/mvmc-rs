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

## Thread sweep, Rayon only (issue #479, `results/inner479_inner{1,4,8,16}.csv`)

Diagnosis (L64 optimization, `MVMC_RS_INNER_PROFILE=1`, 8 workers, gate bypassed): the Rayon pool paid 15 to 30 us
per region and the Pfaffian kernels built per-block tables. Pooled `UpdateMAll` (8 planes, 31 us serial) took 26 us and
`CalculateNewPfM2` (0.8 us serial) 7.3 us, so only the Pfaffian region (847 us serial, 300 us pooled) paid off, and the
size gate (`n_size >= 138` at 8 workers) rightly kept everything serial. The dispatch micro-benchmark
(`tests/rayon_dispatch_479.rs`, ignored, 8 workers, quiet host, per empty 8-block region) separates the causes:

| variant | gap 0 | gap 20 us | gap 60 us |
|---|---|---|---|
| outside the pool: `install` + `scope` | 7.1 us | 17.8 us | 33.2 us |
| outside the pool: `par_iter` + `with_min_len(1)` | 8.3 us | 14.6 us | 30.2 us |
| outside the pool: `ThreadPool::broadcast` | 4.7 us | 4.3 us | 25.5 us |
| inside one hoisted `install`: `scope` | 3.1 us | 6.6 us | 19.9 us |
| inside one hoisted `install`: `broadcast` | 2.8 us | 2.8 us | 18.6 us |
| (spin-pool prototype, replaced) | 1.4 us | 1.3 us | n/a |

Two effects: (1) the caller of `install`/`broadcast` from outside sleeps on a latch, which costs a wake-up per region;
a driver that runs inside the pool joins by running or stealing jobs and never sleeps; (2) Rayon's workers spin only
about 30 us, then sleep, and a region after a longer serial phase pays 19 to 33 us in the micro-benchmark and 100 us or
more in the real kernels (core wake-up). Changes made within Rayon: static blocks with one `broadcast` per region (no
`par_iter` splitting, one scratch per block), the Pfaffian kernels on a reusable staging buffer instead of per-block
tables (keeping the "a failed parallel region leaves the table untouched" boundary), nested regions run inline, a 64 MiB
worker stack so that the `mvmc` binary can run its whole driver inside the pool (`install` in `main`, only when
`MVMC_RS_INNER_THREADS > 1`), and gates that distinguish the context: outside the pool the Rayon-calibrated 100 us work
gate and `n_size` gate are unchanged; inside it 40 us for the sampler loop (regions every few us, workers awake), 400 us for
everything else (workers asleep), no `n_size` gate. Results are byte-identical for 1, 2, 3, 4 and 8 workers
(`threaded_issue182/360/361` and the byte comparison of every output file on Hubbard L16/L32/L64, DH, Kondo, RBM+OptTrans,
Heisenberg complex and FSZ, PhysCal L16, also with the gates bypassed), and the default one-thread outputs are
byte-identical to `main` with the same timing (`results/ab_inner479_1thread_before_after.csv`, 0.99x to 1.03x).

Seconds, median of 2, Rust and native C interleaved (host load before/after every run is in the CSVs; the 4T runs saw
median load 2.6, 8T 4.9, 16T 10.2, so those columns are pessimistic for Rust and C alike):

| workload | Rust 1T | 4T | 8T | 16T | C 1T | C 4T | C 8T | C 16T |
|---|---|---|---|---|---|---|---|---|
| opt_hubbard_L64 | 6.26 | 3.63 | 3.11 | 3.45 | 7.05 | 2.90 | 2.36 | 2.70 |
| opt_hubbard_L32 | 3.68 | 3.69 | 3.70 | 3.71 | 4.45 | 2.51 | 2.46 | 3.26 |
| phys_hubbard_L32 | 10.38 | 10.46 | 10.42 | 10.49 | 16.14 | 7.90 | 7.42 | 8.34 |
| opt_heisenberg_cmp | 13.73 | 13.81 | 13.68 | 13.77 | 18.06 | 14.77 | 17.59 | 23.30 |
| opt_heisenberg_fsz | 4.10 | 4.20 | 3.93 | 3.85 | 4.71 | 8.26 | 11.12 | 16.53 |
| opt_hubbard_rbm_opttrans | 8.27 | 8.04 | 7.58 | 7.39 | 11.26 | 10.68 | 13.17 | 20.62 |
| opt_kondo_real | 7.49 | 7.72 | 7.71 | 7.79 | 9.79 | 12.09 | 15.44 | 19.73 |

Achieved: L64 optimization 1.7x at 4 workers and 2.0x at 8 (C's OpenMP 2.4x and 3.0x; a throw-away spin-pool prototype
reached 2.2x and 2.7x: 2.81 s and 2.29 s in the first version of this PR). The remaining Rayon gap is the
sleep/wake behavior: Rayon offers no spin-window control, so the 3 us hot dispatch is only reachable for regions that
follow each other within 30 us, i.e. the sampler loop. Not reachable with Rayon: PhysCal at 32 sites (the prototype
reached 1.6x; there the regions follow serial phases of ms, the workers are asleep, and pooled regions lose, so the
400 us gate keeps them serial: 10.4 s at every thread count) and the 32-site optimization (its `UpdateMAll` regions are
8 us serial, below the 40 us gate; with `MVMC_RS_INNER_MIN_WORK_NS=10000` a hoisted 8-worker run takes 2.4 s, but that gate also pools the 11 us Kondo Pfaffian regions and the RBM kernels, which lose). The small models stay
within 3 % of their 1T times, where C's OpenMP gets slower with threads (FSZ 3.9x at 16 threads).

