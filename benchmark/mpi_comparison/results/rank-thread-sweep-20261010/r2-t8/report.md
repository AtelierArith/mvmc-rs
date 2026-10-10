# C / Julia / Rust MPI benchmark

Periodic half-filled Hubbard chain: t=1, U=4, Lsub=4, NSPGaussLeg=8.
MPI ranks: 2; compute threads/rank: 8; BLAS threads: 1.
Saved configurations: 320 total per step/group (160/rank).
Opt: 300 SR steps; PhysCal: 100 measurement groups.
Warmups: 1; measured repetitions: 3; times are medians in seconds.
PhysCal uses the same C-generated parameter file for all implementations.

Rust measurement path: calc-m-all (production default C-order kernels).

| Workload | Sites | C (s) | Julia (s) | Rust (s) | Julia/C | Rust/C |
|---|---:|---:|---:|---:|---:|---:|
| Opt | 32 | 5.859710 | 6.012362 | 4.739087 | 1.026 | 0.809 |
| Opt | 64 | 17.635230 | 19.715454 | 18.285619 | 1.118 | 1.037 |
| PhysCal | 32 | 2.219950 | 2.339800 | 2.095023 | 1.054 | 0.944 |
| PhysCal | 64 | 8.237540 | 8.250117 | 7.557637 | 1.002 | 0.917 |

C: fresh-process rank-zero internal All timer, after MPI_Init.
Julia/Rust: warmed production API maximum over ranks, including parsing,
initialization/parameter loading, sampling, optimization or observables, and output.
Process startup, JIT, build/setup and warmups are excluded. Timing boundaries differ.
Raw measurements, outputs, input hashes, source hashes, versions and linked libraries
are retained beside this report. Timing runs are sequential. Energies are not parity proof.
