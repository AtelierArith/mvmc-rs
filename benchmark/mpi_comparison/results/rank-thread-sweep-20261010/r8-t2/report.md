# C / Julia / Rust MPI benchmark

Periodic half-filled Hubbard chain: t=1, U=4, Lsub=4, NSPGaussLeg=8.
MPI ranks: 8; compute threads/rank: 2; BLAS threads: 1.
Saved configurations: 320 total per step/group (40/rank).
Opt: 300 SR steps; PhysCal: 100 measurement groups.
Warmups: 1; measured repetitions: 3; times are medians in seconds.
PhysCal uses the same C-generated parameter file for all implementations.

Rust measurement path: calc-m-all (production default C-order kernels).

| Workload | Sites | C (s) | Julia (s) | Rust (s) | Julia/C | Rust/C |
|---|---:|---:|---:|---:|---:|---:|
| Opt | 32 | 3.023370 | 3.007437 | 2.949700 | 0.995 | 0.976 |
| Opt | 64 | 13.514840 | 12.410990 | 12.994745 | 0.918 | 0.962 |
| PhysCal | 32 | 1.289130 | 1.281828 | 1.095543 | 0.994 | 0.850 |
| PhysCal | 64 | 6.386400 | 5.136011 | 4.853119 | 0.804 | 0.760 |

C: fresh-process rank-zero internal All timer, after MPI_Init.
Julia/Rust: warmed production API maximum over ranks, including parsing,
initialization/parameter loading, sampling, optimization or observables, and output.
Process startup, JIT, build/setup and warmups are excluded. Timing boundaries differ.
Raw measurements, outputs, input hashes, source hashes, versions and linked libraries
are retained beside this report. Timing runs are sequential. Energies are not parity proof.
