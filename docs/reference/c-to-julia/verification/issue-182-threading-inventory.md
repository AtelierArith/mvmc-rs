# Issue #182 inner-threading inventory

Scope: every shared-memory (inner) threading site in `extern/Julia-mVMC`, the
corresponding C OpenMP behavior in `extern/mVMC-1.3.0`, and the Rust
counterpart. InterAll and its Lanczos work are owned separately and excluded.
C remains authoritative for numerical contracts; Julia is the feature
reference. This is a source audit plus a Rust benchmark; no new Julia
numerical run is claimed.

Revisions: Rust `36b1eb5a` (branch base); Julia-mVMC line numbers refer to
`MVMCOptimizers.jl/src/` (identical `src/` tree in the pinned submodule
`c0788c34` and in the local checkout named by the issue, `8bb1b9e8`); mVMC 1.3.0 `src/mVMC/`.

## Key structural findings

1. **The sample loop is not threaded in C, Julia or Rust.** C
   `vmccal.c:114` / `:351` iterate `for(sample=sampleStart;...)` serially after
   `SplitLoop` over MPI ranks. Julia's `threading.jl` header says the same
   ("`VMCMainCal` keeps the C rank-local sample loop sequential"). Parallelizing
   it would change RNG draw order and per-sample state, so it is out of scope
   and must not be attempted under #182.
2. Julia has **14** `Threads.@threads` sites, no `@spawn`/`@sync`, no explicit
   `BLAS.set_num_threads`, and two thread-local buffer families. All are gated
   by `JULIA_MVMC_INNER_THREADS=1` and `Threads.nthreads() > 1`; the default is
   serial.
3. C has 36 files with OpenMP, 365 `omp` pragma lines in `src/mVMC` (parallel, for, master, barrier, critical), almost all on index
   loops *inside* one sample (Green function, Hamiltonian terms, projection,
   Pfaffian update, SR/CG vector kernels). Julia threads only a small subset;
   Rust threads a superset of Julia's set. The remaining C regions are serial in
   both ports (section "C-only regions").

## Julia call sites

Class legend: **EQ** Rust-equivalent (code + test), **BE** backend/policy
difference (documented, intentional), **OUT** out of scope or not applicable.

| # | Julia site | Work | C reference | Rust counterpart | Class |
|---|---|---|---|---|---|
| 1 | `calculate_m_all.jl:445` `@threads for qpidx` (complex) | independent QP Pfaffian/inverse | `matrix.c:96-113` `omp for qpidx` + `critical` (info) | `pfaffian.rs:418` `calc_m_all_complex_with_kernel` (gate `:439-442`, `install` `:468`) | EQ |
| 2 | `calculate_m_all.jl:884` `@threads for qpidx` (real) | same, real | `matrix.c:197-212` | `pfaffian.rs:190` `calc_m_all_real_with_status` (gate `:211-214`, `install` `:240`) | EQ |
| 3 | `threading.jl:66` `copy_real_to_complex!` | disjoint element copy | `vmccal_fsz.c:81` | `threading.rs` `copy_real_to_complex`; callers `run.rs:2025-2033`, `:2696-2701`, `pfaffian.rs` FSZ-real path | EQ |
| 4 | `threading.jl:86` `copy_complex_realpart!` | disjoint element copy | (C copies whole buffers, `vmccal_fsz.c:81`) | `threading.rs` `copy_complex_realpart`; caller `pfaffian.rs:648` | EQ |
| 5 | `vmc_main_cal.jl:1707` `@threads :static for k` (Transfer terms, real fast path, per-`tid` accumulators) | independent Green terms | `calham.c:80-135` `omp for ... reduction(+:e)` | `observables.rs:2818-2860` `par_iter().map_init(GreenScratch)`, then serial in-order reduction | EQ (see differences) |
| 6 | `vmc_main_cal.jl:3403` `calculate_oo!` first block | disjoint `OO/HO` entries | `vmccal.c:773` | `observables.rs:261` `calculate_oo` (`for_each_pair_mut`) | EQ |
| 7 | `vmc_main_cal.jl:3420` `calculate_oo!` rows `i>=2` | disjoint rows | `vmccal.c:783` | `observables.rs:261` `calculate_oo` (`par_chunks_mut`) | EQ |
| 8 | `vmc_main_cal.jl:3459` `calculate_oo_real!` columns | disjoint columns | `vmccal.c` `calculateOO_real` | `observables.rs:219` `calculate_oo_real` | EQ |
| 9 | `vmc_main_cal.jl:3478` `calculate_oo_real!` HO | disjoint | same | `observables.rs:255` | EQ |
| 10 | `vmc_main_cal.jl:3513` `calculate_oo_store!` | disjoint store/HO | `vmccal.c` `calculateOO_Store*` (`:671-720` omp loops for `HO`/mean) | `observables.rs:416` `calculate_oo_store` (`:429`, `:447`) | EQ |
| 11 | `vmc_main_cal.jl:3552` `calculate_oo_store_real!` | disjoint store/HO | same | `observables.rs:318` `calculate_oo_store_real` | EQ |
| 12 | `vmc_main_cal.jl:3587` `finalize_oo_store!` NSRCG diagonal | per-component sample sum | `vmccal.c:707,714` omp mean/diagonal loops | `observables.rs:437` `finalize_oo_store` diagonal-only | EQ |
| 13 | `vmc_main_cal.jl:3627` `finalize_oo_store!` Gram | per-column serial sample sum | `vmccal.c` `calculateOO_Store` ZGEMM (BLAS, not OpenMP; order is BLAS-defined) | `observables.rs:497` `sr_store_gram_julia` (`par_chunks_mut`) | EQ |
| 14 | `vmc_main_cal.jl:3668` `finalize_oo_store_real!` NSRCG diagonal | per-component sum | `vmccal.c:678,714` omp mean/diagonal loops | `observables.rs:338` `finalize_oo_store_real` diagonal-only | EQ |

Non-`@threads` Julia threading-related items:

| Item | Julia | Rust | Class |
|---|---|---|---|
| Thread-local Pfaffian workspaces | `workspace.jl:123,136,159` `ThreadedPfaPackWorkspace`, `get_thread_workspace`, `ensure_thread_capacity!` keyed by `threadid()` | `ThreadedPfaPackWorkspace::take()` per rayon task (`pfaffian.rs:190-280`); scratch isolation test `scratch_isolation` | EQ |
| Thread-local calham1 scratch | `vmc_main_cal.jl:1687-1727` `calh1_thread_*` copies of `ele_idx/ele_num`, per-`tid` accumulator | `GreenScratch` via `map_init` (per-task, no shared mutable state) | EQ |
| Control env | `threading.jl:12-41` `JULIA_MVMC_INNER_THREADS` (0/1), `JULIA_MVMC_PFAPACK_THREADS`; width = `JULIA_NUM_THREADS` | `threading.rs` `MVMC_RS_INNER_THREADS` (width, default 1), `MVMC_RS_INNER_THRESHOLD`; frozen pool (`issue184_thread_control_contracts.rs`) | BE |
| Gate thresholds | `min_work_per_thread=64` (SR/copy), `16` (transfer), `max(.., nthreads)`; QP: serial if `qp_num < nthreads` | one threshold, default 32, `items >= threshold` and `threads > 1` | BE |
| Pfaffian call lock | `threading.jl:33-45` `PFAPACK_CALL_LOCK` (`JULIA_MVMC_PFAPACK_LOCK`) | none: Rust workspaces are per-task and the Pfaffian kernels are pure Rust | OUT (Julia-specific) |
| BLAS threads | no `BLAS.set_num_threads`; BLAS uses Julia/OpenBLAS defaults (SYRK/GEMM in `finalize_oo_store_real!` full Gram, `vmc_sampling.jl:1851-1866` rank-1 updates) | `serial_blas.rs` pins `openblas_set_num_threads(1)` for deterministic order; inner Rust threads never nest BLAS threads | BE |
| MPI interaction | rank-local kernels; sample loop serial per rank | per-rank inner workers; `issue182_mpi_inner_physcal.rs` (feature `mpi`) one worker setting per native process | EQ |

### Documented differences (not gaps)

* **Reduction order.** Julia sums per-`tid` partials in thread order, so the
  result depends on `nthreads()` and the static schedule. Rust computes each
  Green term in parallel into an indexed vector and reduces serially in term
  order, so results are bitwise independent of the worker count. C's
  `reduction(+:e)` order is unspecified; with one thread it equals the serial
  order, which Rust reproduces for every worker count.
* **Gram/copy kernels** have exactly one producer per output cell and serial
  inner reductions, so they are bitwise worker-independent.
* **Real transfer gate.** Both ports parallelize Transfer only on the real
  fast path; Rust additionally skips it when `calham1` diagnostics timers are
  on (matching Julia's `!ctimer_enabled`).
* **Rust supersets.** Rust also threads FSZ-complex `CalculateMAll`
  (`pfaffian.rs:509`) and Green measurement cells
  (`green_measurements.rs` `collect_green_values`, C `calgrn.c`), which Julia
  leaves serial.

## Verification coverage in Rust

All in `crates/mvmc-core/tests/` unless noted. Worker sets are 1/2/4.

* `threaded_issue182.rs`: QP threshold boundaries and scratch isolation
  (`qp_threshold_workers_scratch_and_lifecycle_match_independent_expectations`),
  observation of real dispatch (`observation_classifies_actual_work_without_changing_kernel_records`),
  Transfer boundary (`transfer_site_executes_actual_jobs_at_threshold`), full
  runner with direct store, CG and PhysCal
  (`runner_workers_preserve_rng_configurations_direct_store_cg_and_physcal`),
  20-step repeatability, non-SPD SR boundary, independent PhysCal ordering and
  real/FSZ (`independent_real_fsz_workers_match_public_pre_sr_and_rng`).
* `issue184_thread_copy_contracts.rs`: copy prefix/tail contracts per worker.
* `issue184_thread_control_contracts.rs`: env controls and frozen pool.
* `issue182_mpi_inner_physcal.rs`, `issue182_s181_local_sr.rs`: MPI and
  rank-local SR with inner workers.
* `threading.rs` unit tests: observation scope isolation.

## Counts

| Class | Julia `@threads` sites | Other Julia items |
|---|---|---|
| EQ (Rust-equivalent) | 14 | 3 (thread workspaces, calham1 scratch, MPI interaction) |
| BE (backend/policy difference) | 0 | 3 (env controls, thresholds, BLAS threads) |
| Rust-serial gap vs Julia | 0 | 0 |
| OUT (Julia-specific/not applicable) | 0 | 1 (`PFAPACK_CALL_LOCK`) |

There is **no Julia-threaded site that is serial in Rust**. Gaps exist only
against C (below).

## C-only regions (serial in both Julia and Rust)

These are C OpenMP regions with no Julia counterpart. #182 recorded them; **#360
ports the priority regions** to the same dedicated Rayon pool
(`crates/mvmc-core/src/threading.rs`, `MVMC_RS_INNER_THREADS` /
`MVMC_RS_INNER_THRESHOLD`). Each parallelizes an index loop *within* one sample;
none involves RNG draws. The sample loop itself stays serial. Every ported region
keeps its serial code for one worker (or work below the threshold) and reduces in
fixed index order (per-term/per-element values are computed concurrently, then
combined serially in index order), so results are bit-identical for any worker
count and equal to C's one-thread order.

| C region | Files | Status (#360) | Rust |
|---|---|---|---|
| Diagonal and two-body Hamiltonian terms (CoulombIntra/Inter, Hund, PairHop, Exchange, InterAll) | `calham*.c` | ported (pooled term values, serial in-order sum) | `observables.rs` `calculate_hamiltonian_diagonal`, `calculate_local_energy_timed` via `threading::collect_terms` |
| Lanczos Hamiltonian/Green terms (`lslocgrn*.c`, Lanczos `calham*.c`) | `lslocgrn*.c` | ported (#360 follow-up): terms run concurrently on per-task copies of the Slater matrices/Transfer cache (restored after every term, as the serial evaluator restores the shared state), per-term moved-configuration sums stay serial, values are stored/added in term order | `observables.rs` `calculate_lanczos_green`, `calculate_lanczos_h2_transfer` via `collect_terms` and `lanczos_task_state` |
| Green function accumulation | `calgrn*.c` | already pooled (superset of Julia) | `green_measurements.rs` |
| Sampler updates | `pfupdate*.c` (`updateMAll*`, `calculateNewPfM*`) | ported: QP loops of 15 flat kernels | `sampling/updates.rs` via `threading::qp_fill` / `qp_update` |
| Projection | `projection.c` DH2/DH4 loops | ported (definitions concurrent, integer increments applied in definition order) | `sampling/projection.rs` `recompute_dh_counts` |
| RBM | `rbm.c` `LogWeightRBM`, `RBMRatio` | ported (per-hidden-unit values concurrent, serial `z`/`zz` fold in unit order); `set_rbm_diff` computes each hidden counter's `tanh` concurrently and keeps the term-order accumulation serial; the counter build (`MakeRBMCnt`) is serial in C too | `sampling/rbm.rs` `log_rbm_val`, `log_rbm_ratio` |
| Slater setup | `slater*.c` `UpdateSlaterElm*` | ported (one producer per QP plane, real shadow copy elementwise) | `slater_update.rs` |
| SR/CG vector kernels | `stcopt.c`, `stcopt_cg_impl.c` | ported: S diagonal, S/g assembly (per column), CG axpy/residual/direction/product-correction loops and the sample-matrix fill; dot products and GEMV stay serial by design (Julia/C operation order) | `sr.rs`, `sr_cg.rs` |
| `average.c` | `average.c` | ported: the `1/Wc` scaling of the SR/energy buffers | `average.rs` |
| `stcopt_pdposv.c` | | not applicable: it distributes the S matrix for ScaLAPACK `pdposv`; Rust factors with the serial-pinned LAPACK `dpotrf`/`dpotrs` (fixed operation order) and already pools the S/g assembly | `sr.rs` |
| `parameter.c` | | not ported: `InitParameter` is one-time initialization whose loops interleave with RNG draws, and the zeroing/rescale loops live in `mvmc-expert-parsers`, which has no inner pool; `update_parameter_value` mutates `ExpertModeData` sequentially | |
| Setup | `qp.c`, `gauleg.c`, `workspace.c`, `matrix.c` setup | not ported: one-time work per run (QP weights, Gauss-Legendre nodes, workspace allocation) whose cost does not scale with the sample or step count; the `matrix.c` QP loop is the already-pooled `calc_m_all_*` | |

Proof: `crates/mvmc-core/tests/threaded_issue360.rs` runs every worker setting in
its own process (threshold 1) and requires identical bits for the QP kernels
(including start offsets and the coincident-index branch) and byte-identical
output files for short optimizations (Gutzwiller/Jastrow/DH, direct SR and
NSRCG with stored samples, RBM with OptTrans, all Hamiltonian terms), plus the
observed pooled-dispatch counters for 2 and 4 workers and none for one worker.
InterAll is exercised by dedicated cases that append an InterAll definition to the
Hubbard fixtures (optimization with direct SR and CG, and PhysCal); the Lanczos
Transfer/PairHop/Exchange terms and the Lanczos Green terms by the Lanczos 1 and 2
PhysCal fixtures.

## Benchmark

Purpose: record the effect of the existing inner threading on the report
Hubbard-chain inputs. No speed claim is made. `xtask bench-hubbard --threads N`
did not set `MVMC_RS_INNER_THREADS` when this was measured (it now takes `--inner-workers`, #361; see
`benchmark/hubbard_chain/results/hubbard_chain_2026-10-06_inner_threads.md`);
the runs below invoke the release CLI directly.

Settings: `target/release/mvmc benchmark/hubbard_chain/inputs/hubbard_chain_L{16,32}/namelist.def
--mode real --nsteps 300 --nsmp 300`, one run per cell, `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=N`, `MVMC_RS_INNER_THREADS=N`,
`MVMC_RS_INNER_THRESHOLD` as listed, `RndSeed=1`, MPI ranks 1.

Environment: Linux 6.8 x86_64, Intel Xeon E5-2699 v3 (36 logical CPUs, shared
host with load average about 14-18 during the runs), rustc 1.99.0, Rust commit
`36b1eb5a`, release profile, OpenBLAS (Rust link, pinned to 1 thread by
`serial_blas.rs`). Julia was not run.

| L | threshold | workers 1 | workers 2 | workers 4 | final energy/site (all workers) |
|---|---|---|---|---|---|
| 16 | 32 (default) | 19.5 s | 37.6 s | 30.6 s | -0.5418807043 |
| 16 | 2 | 15.1 s | 37.5 s | 36.6 s | -0.5418807043 |
| 32 | 32 (default) | 76.4 s | 92.0 s | 80.2 s | -0.4962523984 |
| 32 | 2 | 55.4 s | 102.7 s | 93.1 s | -0.4962523984 |

Findings:

* The final energy and the last `zvo_out.dat` row agree for 1/2/4 workers
  (deterministic reductions); the correctness contract is covered by the tests
  above, not by this table.
* Wall times are noisy: the two `workers=1` L=16 runs differ by 30% for
  configurations that dispatch identical (serial) work, because the host was
  shared. Treat differences under about 30% as noise.
* Enabling inner workers on these small problems (QP count and SR size in the
  tens) is **slower or neutral**, not faster. Thread dispatch cost exceeds the
  per-kernel work; the default (`MVMC_RS_INNER_THREADS=1`) remains the right
  setting for the report inputs. A speedup claim would need larger SR/QP sizes
  and a quiet host; none is made.
* No allocation or timer counters beyond the existing observation counters
  (`threading::start_observation`) were collected.
