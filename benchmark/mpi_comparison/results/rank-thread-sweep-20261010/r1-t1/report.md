# C / Julia / Rust MPI benchmark

Periodic half-filled Hubbard chain: t=1, U=4, Lsub=4, NSPGaussLeg=8.
MPI ranks: 1; compute threads/rank: 1; BLAS threads: 1.
Saved configurations: 320 total per step/group (320/rank).
Opt: 300 SR steps; PhysCal: 100 measurement groups.
Warmups: 1; measured repetitions: 3; times are medians in seconds.
PhysCal uses the same C-generated parameter file for all implementations.

Rust measurement path: calc-m-all (production default C-order kernels).

| Workload | Sites | C (s) | Julia (s) | Rust (s) | Julia/C | Rust/C |
|---|---:|---:|---:|---:|---:|---:|
| Opt | 32 | 15.528920 | 11.522847 | 11.964399 | 0.742 | 0.770 |
| Opt | 64 | 69.061990 | 50.136607 | 56.532043 | 0.726 | 0.819 |
| PhysCal | 32 | 7.464570 | 4.515244 | 4.657781 | 0.605 | 0.624 |
| PhysCal | 64 | 39.351040 | 19.473849 | 22.703444 | 0.495 | 0.577 |

C: fresh-process rank-zero internal All timer, after MPI_Init.
Julia/Rust: warmed production API maximum over ranks, including parsing,
initialization/parameter loading, sampling, optimization or observables, and output.
Process startup, JIT, build/setup and warmups are excluded. Timing boundaries differ.
Raw measurements, outputs, input hashes, source hashes, versions and linked libraries
are retained beside this report. Timing runs are sequential. Energies are not parity proof.
