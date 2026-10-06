# CalculateMAll (Pfaffian setup) gap on the Hubbard real path (issue #415)

Related to #185, #207. Follows `hubbard_chain_2026-10-06_calh1_slaterdiff.md` (PR #414), which left
`CalculateMAll` at 1.06-1.21x the native C section time.

## Summary

* `CalculateMAll [40]` is now **0.80x / 0.86x / 0.88x** of native C at 16 / 24 / 32 sites (it
  was 1.08x / 1.19x / 1.20x in the interleaved baseline of this run), and `recal PfM and
  InvM [34]` is 0.82x / 0.89x / 0.93x of C. Every optimization output is **bit-identical** to
  `origin/main` `dfa9515f` (`zvo_out.dat`, `zqp_opt.dat` digests and final energy equal in all
  15 interleaved 300-step cells), so the C operation order, RNG draws and acceptance decisions
  are untouched and no tolerance was changed.
* 300 SR steps (internal run time): 0.82x / 0.83x / 0.82x of the previous Rust time
  (8.63 -> 7.08 s, 18.88 -> 15.58 s, 34.90 -> 28.73 s medians).
* Section `All [0]` is 0.75x / 0.82x / 0.85x of native C (20 steps).

## Environment

* Linux 6.8 x86_64, Intel Xeon E5-2699 v3 (shared host). The 20-step section runs saw host
  load1 of 1.5-2.7 (other jobs were running; interleaving makes the C/before/after cells of a
  rep see the same load); the 300-step runs saw 1.0-1.25.
* rustc 1.99.0, release profile (`opt-level=3`, thin LTO, `codegen-units=1`, `debug=1`),
  OpenBLAS 0.3.26 pthread, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, 1 MPI rank,
  `MVMC_RS_INNER_THREADS` unset (except the workers check below).
* C: `extern/mVMC-1.3.0` `vmc.out`, plain `Release` (`-O3`, gcc 13.3.0), built in the Dev
  Container image (MPICH, container OpenBLAS), `mpiexec -n 1`, same inputs with only
  `NSROptItrStep`/`NSROptItrSmp` set to 20 for the section runs; section times are the C
  `zvo_CalcTimer.dat`.
* "before" is `origin/main` `dfa9515f`, "after" is this change; both release binaries were run
  **interleaved** (`rep -> L -> C, before, after` for sections, `rep -> L -> before, after`
  for 300 steps), 5 reps. Raw rows: `hubbard_chain_2026-10-06_calculatemall_sections.csv`,
  `hubbard_chain_2026-10-06_calculatemall_full300.csv`,
  `hubbard_chain_2026-10-06_calculatemall_inner_workers.csv`.

## Section timers (20 SR steps, median (min-max) of 5, seconds)

| sites | section | C (native) | Rust before | Rust after | after/before | after/C |
|---:|---|---:|---:|---:|---:|---:|
| 16 | `CalculateMAll [40]` | 0.302 (0.293-0.316) | 0.328 (0.327-0.341) | 0.240 (0.238-0.242) | 0.73 | 0.80 |
| 16 | `recal PfM and InvM [34]` | 0.105 (0.103-0.111) | 0.119 (0.117-0.123) | 0.086 (0.086-0.089) | 0.72 | 0.82 |
| 16 | `VMCMainCal [4]` | 0.430 (0.415-0.452) | 0.425 (0.424-0.442) | 0.337 (0.333-0.338) | 0.79 | 0.78 |
| 16 | `All [0]` | 0.830 (0.805-0.881) | 0.750 (0.744-0.778) | 0.627 (0.620-0.635) | 0.84 | 0.75 |
| 24 | `CalculateMAll [40]` | 0.631 (0.626-0.644) | 0.749 (0.736-0.756) | 0.545 (0.543-0.550) | 0.73 | 0.86 |
| 24 | `recal PfM and InvM [34]` | 0.216 (0.213-0.222) | 0.263 (0.261-0.269) | 0.192 (0.189-0.193) | 0.73 | 0.89 |
| 24 | `VMCMainCal [4]` | 0.853 (0.850-0.872) | 0.933 (0.917-0.944) | 0.725 (0.721-0.730) | 0.78 | 0.85 |
| 24 | `All [0]` | 1.660 (1.649-1.708) | 1.643 (1.620-1.668) | 1.354 (1.346-1.362) | 0.82 | 0.82 |
| 32 | `CalculateMAll [40]` | 1.121 (1.115-1.139) | 1.347 (1.338-1.361) | 0.989 (0.987-1.003) | 0.73 | 0.88 |
| 32 | `recal PfM and InvM [34]` | 0.379 (0.376-0.385) | 0.481 (0.480-0.484) | 0.352 (0.348-0.358) | 0.73 | 0.93 |
| 32 | `VMCMainCal [4]` | 1.476 (1.466-1.504) | 1.655 (1.635-1.677) | 1.288 (1.285-1.313) | 0.78 | 0.87 |
| 32 | `All [0]` | 2.926 (2.901-2.957) | 2.993 (2.961-3.007) | 2.493 (2.471-2.530) | 0.83 | 0.85 |

## 300 SR steps, internal run time (median (min-max) of 5 interleaved reps, seconds)

| sites | before | after | after/before | host load1 | output digests (zvo_out, zqp_opt, final energy) |
|---:|---:|---:|---:|---|---|
| 16 | 8.63 (8.61-8.69) | 7.08 (7.08-7.16) | 0.82 | 1.01-1.25 | identical in 5/5 |
| 24 | 18.88 (18.73-19.02) | 15.58 (15.53-15.70) | 0.83 | 1.00-1.18 | identical in 5/5 |
| 32 | 34.90 (34.64-34.98) | 28.73 (28.60-29.67) | 0.82 | 1.00-1.21 | identical in 5/5 |

## Inner-thread path (`MVMC_RS_INNER_THREADS` = 1/2/4, 30 steps, 3 interleaved reps)

At these matrix sizes the #360/#361 work gate keeps `CalculateMAll` serial for every worker
count (times are flat across 1/2/4 workers, before and after), so the per-QP kernels are the
only thing that changed. After/before is 0.82-0.84 at 16 sites (1.04 -> 0.86 s) and 0.83 at 32
sites (4.25 -> 3.55 s) for every worker count, and `zvo_out.dat`/`zqp_opt.dat` digests are
identical between before/after and across worker counts in all 36 cells. The multi-worker
chunk path (which calls the same `calc_m_all_child_real`) is exercised by the existing
inner-thread bit-identity tests in `mvmc-core`.

## Profile and what changed (32 sites, one QP plane, n = 32, microseconds)

Per-plane timers (temporary instrumentation, 64944 planes over 20 steps):

| phase | before | after |
|---|---:|---:|
| Slater assembly | 4.9 | 1.5 (assembly + all-zero check fused) |
| all-zero check (`frobenius_norm_sqr_real`) | 0.5 | fused |
| `dsktf2` + `utu2pfa` | 7.3 | 6.0 (5.0 in an isolated loop) |
| `utu2inv`: `dtrtri` | 4.9 | 4.9 (unchanged, BLAS) |
| `utu2inv`: skew-tridiagonal solve | 6.6 | about 3.5 |
| `utu2inv`: `dtrmm` | 2.1 | 2.1 (unchanged, BLAS) |
| `utu2inv`: fill, lacpy, vT, permutations | 2.2 | about 1.5 |
| final sign flip | 0.25 | 0.25 |

Changes (value-affecting operation order untouched; every element is computed by the same
sequence of IEEE operations as before):

* **Slater assembly** (`assemble_inv_m_real`): the spin-shifted site indices depend only on the
  electron slot, so they are resolved and validated once per plane instead of per matrix
  entry (the inner loop was a per-entry `i64` index computation plus two range checks). The
  all-zero check maximum is accumulated in the same pass, in the same column-major order. An
  invalid site falls back to the previous incremental loop (same error, same partial writes).
* **`dsktf2`**: pivot search and row/column swaps work on the column-major slice instead of
  per-element `get`/`set`; the real `DSKR2` update `(a + x*t1) - y*t2` (C order, no FMA)
  runs four lanes at a time with AVX2 (masked tail, so no element is touched twice) when the
  CPU supports it and `blas-backend` is on, with the unchanged scalar loop as the fallback;
  `dscal` of the pivot column is a plain multiply loop for finite nonzero alpha (the same
  products; the BLAS call is kept for zero/non-finite alpha).
* **`utu2inv` (C order, real)**: the skew-tridiagonal solve keeps its `x / y` operations but
  advances four independent columns together, so their serial division chains overlap
  (division is not vectorized, only pipelined); the pivot permutations swap whole columns
  with `swap_with_slice` / per-column `swap`.
* `dtrtri` and `dtrmm` stay on the BLAS/LAPACK provider as in C; a Rust replacement would
  not reproduce OpenBLAS's accumulation order (FMA kernels) and was not attempted.

Tests (same-implementation exact checks, not Rust-vs-C float comparisons):
`rank2_c_order_dispatch_matches_scalar_reference_bitwise` (all column-length remainders),
`real_scale_column_matches_plain_multiplication_bitwise`,
`real_c_order_tridiagonal_solve_matches_direct_reference_bitwise` (four-column groups,
remainders, generic default), `real_assembly_fast_path_matches_indexed_reference_bitwise`
(including the fused maximum and the invalid-site fallback). The existing C-/Julia-fixture,
SR/CG, PhysCal and Lanczos suites pass unchanged.

## Limitations

* Single shared host; the section tables are 20-step medians of 5 (ranges under 4%), the C
  numbers come from a container build (gcc 13.3 `-O3`, container OpenBLAS) on the same CPU and
  are a reference for ratios, not C's best build.
* Real-mode Hubbard-chain inputs only. The AVX2 kernel is x86_64 plus `blas-backend` only; the
  complex and FSZ paths share only the generic pivot-search/swap/permutation edits (no
  complex-specific tuning). On other CPUs the scalar loops run as before.
