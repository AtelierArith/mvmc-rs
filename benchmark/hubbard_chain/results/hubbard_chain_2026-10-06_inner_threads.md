# Inner-threading slowdown: diagnosis, fix and re-measurement (issue #361)

Related to #185, #360 (C OpenMP regions ported to the inner Rayon pool) and #182
(first measurement, `issue-182-threading-inventory.md`).

## Summary

* Before the fix, enabling inner workers made the Hubbard chain **slower at every
  size up to 192 sites** (4 workers: 2.1x slower at 16 sites, 1.9x at 32, 1.17x at
  64) and neutral at 256. After the fix, 2 and 4 workers are within 3% of one worker
  (median) at 16-128 sites, 4 workers are 14% faster at 192 sites and 2 and 4 workers
  are 19% and 37% faster at 256 sites.
* The slowdown has two causes, neither of them BLAS: (1) the Rayon pool was
  dispatched for regions that run in 1-30 us serially (60,000 dispatches in 20 steps
  at 16 sites, about 17-25 us each); (2) even regions with enough work lose, because a
  woken worker runs 2-3x slower than the same code on the main thread (placement and
  Rayon idle-spin on a shared host) and the serial kernels that follow a pooled region
  slow down by 1.4-2.8x (cache lines modified by the workers). Pooling *some* regions
  loses more than pooling all or none.
* The fix is a gate, not a faster pool: without `MVMC_RS_INNER_THRESHOLD`, a region is
  pooled only when the electron matrices are large (`n_size >= 120*w/(w-1)`, i.e.
  160 for 4 workers) **and** its estimated work is at least 100 us. All worker counts
  remain bit-identical (`zvo_out.dat` of every cell below has the same digest for 1, 2
  and 4 workers and for both binaries).
* Acceptance "2/4 workers never slower than 1 beyond noise; faster at the largest
  size": met for the sizes measured (16-256 sites). The speed-up exists only for
  matrices of 192 sites and above; the report inputs (L=16/24/32) and L=64 run
  serially by design, so inner workers give them no benefit. Julia 1.13.1 comparison:
  see the end.

## Environment

* Host: shared Linux 6.8 x86_64, Intel Xeon E5-2699 v3 (18 cores, 36 logical CPUs,
  2.3 GHz nominal), other agent sessions compiling on the same host. Load average
  (1 min) during the measurements is in the last column of each table (1.0-2.1; it
  reached 3.9 only during the first, superseded run of L=16).
* rustc 1.99.0 (2026-09-28), release profile (`opt-level=3`, thin LTO, `debug=1`).
* BLAS: OpenBLAS 0.3.26 pthread build (`/usr/lib/x86_64-linux-gnu/openblas-pthread`),
  `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `BLIS_NUM_THREADS=1` for every run (the Rust side additionally pins BLAS to one
  thread, `serial_blas.rs`).
* Julia 1.13.1 with the checked-in `Manifest-v1.13.toml`.
* Code: `origin/main` `632f28f2` (PR #402) for "before"; the same plus this change for
  "after". `before` is run with `MVMC_RS_INNER_THRESHOLD=32` (the old default), `after`
  with the threshold unset (automatic gate). Runs alternate
  `rep -> workers (1, 2, 4) -> binary (before, after)`, so slow phases of the host hit
  all cells alike.
* Workload: `mvmc <namelist.def> --mode real --nsteps S --nsmp S`, RndSeed=1,
  1 MPI rank, Hubbard chain at half filling (`n_size` = number of sites).
  `S` shrinks with the size so a run takes 1-13 s: 20, 10, 4, 2, 1, 1 steps for
  16, 32, 64, 128, 192, 256 sites. Inputs: `benchmark/hubbard_chain/inputs/`
  for 16, 32, 64; 128, 192, 256 were generated with `mvmc -s` from the same
  `StdFace.def` with `L`/`Ncond` changed (not committed).
* Raw rows: `hubbard_chain_2026-10-06_inner_threads.csv` (wall seconds of the whole
  process, external `date`; the `zvo_out.dat` digest column).

## Results

Cells are `median (min-max)` seconds over the stated repetitions; `2w/1w` and `4w/1w`
are ratios of medians (below 1.00 is faster).

**before (PR #402, threshold 32)**

| sites (n_size) | SR steps | reps | 1 worker | 2 workers | 4 workers | 2w / 1w | 4w / 1w | host load1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 20 | 7 | 1.041 (1.031-1.105) | 1.934 (1.863-2.137) | 2.181 (2.037-2.408) | 1.86 | 2.09 | 1.2-2.1 |
| 32 | 10 | 7 | 2.062 (2.045-2.214) | 3.281 (2.944-4.274) | 3.960 (3.068-4.935) | 1.59 | 1.92 | 1.3-2.1 |
| 64 | 4 | 7 | 3.944 (3.896-3.977) | 4.890 (4.730-4.970) | 4.601 (4.534-4.648) | 1.24 | 1.17 | 1.0-1.3 |
| 128 | 2 | 5 | 3.723 (3.715-3.772) | 4.077 (3.976-4.143) | 3.884 (3.815-3.939) | 1.09 | 1.04 | 1.0-1.6 |
| 192 | 1 | 5 | 5.688 (5.595-5.916) | 5.925 (5.704-5.949) | 5.538 (5.531-5.641) | 1.04 | 0.97 | 1.2-1.6 |
| 256 | 1 | 3 | 13.108 (13.053-13.659) | 13.791 (13.367-13.826) | 13.363 (13.213-13.402) | 1.05 | 1.02 | 1.4-2.1 |

**after (this change, automatic gate)**

| sites (n_size) | SR steps | reps | 1 worker | 2 workers | 4 workers | 2w / 1w | 4w / 1w | host load1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 20 | 7 | 1.028 (1.018-1.079) | 1.042 (1.021-1.297) | 1.045 (1.034-1.281) | 1.01 | 1.02 | 1.2-2.1 |
| 32 | 10 | 7 | 2.072 (2.052-2.109) | 2.116 (2.062-2.319) | 2.117 (2.076-2.326) | 1.02 | 1.02 | 1.3-2.1 |
| 64 | 4 | 7 | 3.946 (3.928-3.969) | 3.984 (3.945-4.202) | 4.021 (3.950-4.109) | 1.01 | 1.02 | 1.0-1.3 |
| 128 | 2 | 5 | 3.762 (3.700-3.811) | 3.866 (3.761-4.009) | 3.742 (3.715-3.887) | 1.03 | 0.99 | 1.0-1.6 |
| 192 | 1 | 5 | 5.559 (5.419-5.670) | 5.615 (5.546-5.917) | 4.754 (4.557-4.838) | 1.01 | 0.86 | 1.2-1.6 |
| 256 | 1 | 3 | 12.847 (12.812-13.466) | 10.453 (10.189-10.553) | 8.142 (7.903-8.367) | 0.81 | 0.63 | 1.4-2.1 |

Reading the table (no claim beyond the medians):

* 16-128 sites: with workers configured, the medians are 0-3% above the serial run.
  The 1-3% is of the order of the run-to-run spread of the serial run itself
  (min-max 1-4%); the larger maxima at 16 sites (1.28-1.30 s) are single outliers
  under host load. Nothing is faster here, by design.
* 192 sites: 4 workers 14% faster (non-overlapping ranges); 2 workers still serial
  (240 needed) and equal to 1 worker.
* 256 sites: 2 workers 19% faster, 4 workers 37% faster, both with non-overlapping
  ranges. An additional single run with 8 workers (interleaved, before the final
  build, same input) took 6.7 s against 13.5 s serial (about 2x); it is not part of
  the tables.
* Bit-identity: for every size the `zvo_out.dat` digest of all 1/2/4-worker runs of both
  binaries is identical (column `zvo_out_md5_8` of the CSV has one value per size).
* `bench-hubbard` through xtask (internal timing, Julia 1.13.1, 20 steps, 3 reps,
  1 warm-up) for the report inputs:

  | model | threads (BLAS/Julia) | Rust inner workers | Rust median (s) | Julia median (s) |
  |---|---:|---:|---:|---:|
  | L16 | 1 | 1 | 1.010 | 0.853 |
  | L32 | 1 | 1 | 4.070 | 3.354 |
  | L16 | 4 | 4 | 1.030 | 0.921 |
  | L32 | 4 | 4 | 4.160 | 3.881 |

  Rust is within 3% of itself with 4 inner workers; Julia is 8-16% slower with 4
  BLAS/Julia threads than with 1 on these tiny matrices. The Rust/Julia ratio is the
  usual 0.82-0.93x of `hubbard_chain_2026-10-03.md`; inner threading does not change
  it for these inputs. Final energies agree to 4.4e-16 between Rust and Julia.

## Diagnosis

Tools added: `MVMC_RS_INNER_PROFILE=1` makes the CLI print, per call site, the number
of serial/pooled calls, items and wall time; `MVMC_C_TIMER=1` gives the section timers.

1. **Dispatch overhead per tiny region.** 16 sites, 20 steps, 4 workers, threshold 32
   (pooled call sites only; "serial" is the same site with one worker):

   | site | pooled calls | pooled us/call | serial us/call |
   |---|---:|---:|---:|
   | `copy_real_to_complex` (`threading.rs`) | 48,000 | 17.1 | 1.5 |
   | `calculate_oo_store_real` accumulate (`observables.rs`) | 6,000 | 15.0 | 0.8 |
   | transfer Green terms (`CalHamiltonian1`) | 6,000 | 24.6 | 16.2 (section timer) |
   | `update_slater_elm` real shadow copy | 20 | 79.8 | 45.6 |

   Together 60,000 dispatches (3,000 per SR step) of 15-25 us each: a region needs
   roughly 100 us of work before a dispatch can pay. Section timers, 16 sites, 20
   steps, 1 vs 4 workers: `All` 1.07 s -> 2.37 s, `VMCMainCal` 0.65 -> 1.85,
   `calculate OO and HO` 0.005 -> 0.106, `LocEnergyCal` 0.099 -> 0.177,
   `CalculateMAll` 0.349 -> 0.442, `hopping update` 0.250 -> 0.314 (the sampling
   regions are serial at this size and still slow down, see 4).
2. **Per-task scratch allocation.** Each pooled `calc_m_all` chunk allocates and zeroes
   an `InvMColMajor` for `end` planes and copies the planes back. Replacing it by a
   reused thread-local buffer did not change the pooled time (410 us per call at
   32 sites either way), so it is not the dominant cost. `map_init` scratch of the
   QP kernels is per Rayon split and small; the Slater/QP kernels have no per-item
   allocation.
3. **BLAS oversubscription: ruled out.** `OPENBLAS_NUM_THREADS=1` everywhere; a
   stand-alone `dsktf2`+`utu2inv` loop on 1, 2 and 4 plain threads (n=32, OpenBLAS
   pthread) takes 30, 29 and 33 us per call: the Pfaffian kernels scale perfectly when
   not sharing data. Four independent `mvmc` processes run only 12% slower than one.
4. **Placement, idle spinning and cross-thread cache effects.** In the 32-site run the
   per-QP kernels take 2.9x longer on pool workers than on the main thread in *every*
   phase (assemble 7.0 -> 18.2 us, factorisation 7.8 -> 23.4, inverse 15.8 -> 46.0,
   final negation 0.3 -> 0.9). `strace -f -c` of a 4-step run shows 113,761
   `sched_yield` calls (Rayon workers spin after every region) for 1,628 pooled
   dispatches, and `ps -L` shows the main thread and the four workers on the same
   CPU or on hyperthread siblings. Pinning the process to four distinct cores
   (`taskset -c 2,4,6,8`) lowers the pooled `calc_m_all` from about 410 to 275 us per
   call (serial 236 us), so placement is a large part, not all, of the loss; the
   workers also pull the `inv_m` planes out of the main thread's cache, so serial
   kernels after a pooled region are slower (`calculate_new_pf_m2` 3.0 -> 8.5 us per
   call at 128 sites, the `real -> complex` copy 92 -> 253 us).
5. **Partial pooling is worse than none.** 128 sites, 4 workers, interleaved, 3 runs
   (2 steps), minimum region work `MVMC_RS_INNER_MIN_WORK_NS`:

   | min work | 100 us (all regions pooled) | 1 ms | 4 ms | 16 ms (none pooled) | serial (1 worker) |
   |---|---:|---:|---:|---:|---:|
   | wall (s) | 3.84, 3.75, 3.79 | 4.45, 4.57, 4.72 | 4.63, 4.69, 4.61 | 3.80, 3.74, 3.99 | 3.69, 3.68, 3.79 |

   Pooling only the Pfaffian setup (6 ms of work per call) while the rest stays serial
   is 17-25% slower than either extreme, which is why a per-region work threshold alone
   cannot be tuned: the gate has to be all-or-none by problem size.

## What changed

* `threading::inner_parallel_work(items, cost_ns)`: with `MVMC_RS_INNER_THRESHOLD`
  set, the old item-count gate (the worker-invariance tests use it to force pooled
  execution of tiny inputs); otherwise pooled iff `items >= 2` and
  `items * cost_ns >= MVMC_RS_INNER_MIN_WORK_NS` (default 100 us).
  Every pooled call site now passes a calibrated per-item cost (fits of the serial
  per-item times printed by `MVMC_RS_INNER_PROFILE=1` at 16/32/64 sites, e.g. Pfaffian
  plane `28 n^2 + n^3/8` ns: 7.4/29.5/145 us measured at n=16/32/64).
* `threading::scaled_cost_ns(n_size, cost)`: the matrix-dependent regions (Pfaffian
  setup, rank-one QP updates, QP fills, Green terms, Slater planes) report cost 0
  (serial) below `MVMC_RS_INNER_MIN_SIZE`, default `120*w/(w-1)` for `w` workers
  (160 for 4, 240 for 2, 138 for 8): crossovers of the table above (4 workers: 128
  neutral, 160 about 10% faster in a 3-run side measurement, 192 14% faster; 2
  workers: 192 equal, 256 19% faster).
* `calc_m_all_real` uses `threading::static_blocks` (Rayon `broadcast`, worker `k`
  always owns block `k`, the C `omp for` static schedule) so a worker keeps its QP
  planes in cache from call to call (pooled call 410 -> 324 us at 32 sites, unpinned).
* `MVMC_RS_INNER_PROFILE=1` per-call-site profile; `xtask bench-hubbard` and
  `bench-julia` options `--inner-workers`, `--inner-threshold`, `--inner-cpus`
  (Rust only; recorded in the report or `<csv>.config.txt`); the 64-site input.
* Results are unchanged: `threaded_issue182`, `threaded_issue360` (now run both with
  the explicit threshold and with the automatic gate forced open) and the new
  `threaded_issue361` pass; the digest check above covers the real runs.

## Limitations

* Single host, shared with other sessions; ranges, not repeated across days. The
  crossover sizes (160/192/240) are measured with the Hubbard chain at half filling
  (`n_size` = sites); other models with the same `n_size` have different per-region
  costs, the gate uses the calibrated estimates only.
* The estimates are per-host fits (Xeon E5-2699 v3); a faster core moves the 100 us
  work gate and the size gate. `MVMC_RS_INNER_MIN_WORK_NS` and `MVMC_RS_INNER_MIN_SIZE`
  override them; `--inner-threshold` forces the old gate.
* 2-3x slower workers at small sizes come from scheduler placement and Rayon idle
  spinning on this host; with pinned cores the loss is smaller (see 4) but the pooled
  `calc_m_all` at 32 sites was still slower than serial (275 vs 236 us per call).
* The 64-site Julia comparison was not run (Julia takes several minutes per run at
  300 steps); 16 and 32 sites above use 20 steps.
