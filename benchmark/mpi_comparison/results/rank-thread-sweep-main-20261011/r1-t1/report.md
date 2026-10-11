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
| Opt | 32 | 15.451220 | 11.563084 | 12.006087 | 0.748 | 0.777 |
| Opt | 64 | 68.305750 | 49.804850 | 55.071552 | 0.729 | 0.806 |
| PhysCal | 32 | 7.424290 | 4.460499 | 4.611857 | 0.601 | 0.621 |
| PhysCal | 64 | 39.265550 | 19.454922 | 22.600687 | 0.495 | 0.576 |

C: fresh-process rank-zero internal All timer, after MPI_Init.
Julia/Rust: warmed production API maximum over ranks, including parsing,
initialization/parameter loading, sampling, optimization or observables, and output.
Process startup, JIT, build/setup and warmups are excluded. Timing boundaries differ.
Raw measurements, outputs, input hashes, source hashes, versions and linked libraries
are retained beside this report. Timing runs are sequential. Energies are not parity proof.
