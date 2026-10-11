# C / Julia / Rust MPI benchmark

Periodic half-filled Hubbard chain: t=1, U=4, Lsub=4, NSPGaussLeg=8.
MPI ranks: 4; compute threads/rank: 4; BLAS threads: 1.
Saved configurations: 320 total per step/group (80/rank).
Opt: 300 SR steps; PhysCal: 100 measurement groups.
Warmups: 1; measured repetitions: 3; times are medians in seconds.
PhysCal uses the same C-generated parameter file for all implementations.

Rust measurement path: calc-m-all (production default C-order kernels).

| Workload | Sites | C (s) | Julia (s) | Rust (s) | Julia/C | Rust/C |
|---|---:|---:|---:|---:|---:|---:|
| Opt | 32 | 3.630270 | 3.418607 | 3.185791 | 0.942 | 0.878 |
| Opt | 64 | 14.106930 | 13.217460 | 13.749864 | 0.937 | 0.975 |
| PhysCal | 32 | 1.536180 | 1.386305 | 1.405597 | 0.902 | 0.915 |
| PhysCal | 64 | 6.877440 | 5.238496 | 5.787171 | 0.762 | 0.841 |

C: fresh-process rank-zero internal All timer, after MPI_Init.
Julia/Rust: warmed production API maximum over ranks, including parsing,
initialization/parameter loading, sampling, optimization or observables, and output.
Process startup, JIT, build/setup and warmups are excluded. Timing boundaries differ.
Raw measurements, outputs, input hashes, source hashes, versions and linked libraries
are retained beside this report. Timing runs are sequential. Energies are not parity proof.
