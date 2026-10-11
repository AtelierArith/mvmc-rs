# C / Julia / Rust MPI benchmark

Periodic half-filled Hubbard chain: t=1, U=4, Lsub=4, NSPGaussLeg=8.
MPI ranks: 16; compute threads/rank: 1; BLAS threads: 1.
Saved configurations: 320 total per step/group (20/rank).
Opt: 300 SR steps; PhysCal: 100 measurement groups.
Warmups: 1; measured repetitions: 3; times are medians in seconds.
PhysCal uses the same C-generated parameter file for all implementations.

Rust measurement path: calc-m-all (production default C-order kernels).

| Workload | Sites | C (s) | Julia (s) | Rust (s) | Julia/C | Rust/C |
|---|---:|---:|---:|---:|---:|---:|
| Opt | 32 | 3.393060 | 3.679634 | 3.455054 | 1.084 | 1.018 |
| Opt | 64 | 16.827430 | 16.429374 | 16.186473 | 0.976 | 0.962 |
| PhysCal | 32 | 1.168410 | 1.462967 | 1.038802 | 1.252 | 0.889 |
| PhysCal | 64 | 6.638500 | 6.086989 | 4.868110 | 0.917 | 0.733 |

C: fresh-process rank-zero internal All timer, after MPI_Init.
Julia/Rust: warmed production API maximum over ranks, including parsing,
initialization/parameter loading, sampling, optimization or observables, and output.
Process startup, JIT, build/setup and warmups are excluded. Timing boundaries differ.
Raw measurements, outputs, input hashes, source hashes, versions and linked libraries
are retained beside this report. Timing runs are sequential. Energies are not parity proof.
