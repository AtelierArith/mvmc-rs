# C / Julia / Rust MPI benchmark

Periodic half-filled Hubbard chain: t=1, U=4, Lsub=4, NSPGaussLeg=8.
MPI ranks: 1; compute threads/rank: 16; BLAS threads: 1.
Saved configurations: 320 total per step/group (320/rank).
Opt: 300 SR steps; PhysCal: 100 measurement groups.
Warmups: 1; measured repetitions: 3; times are medians in seconds.
PhysCal uses the same C-generated parameter file for all implementations.

Rust measurement path: calc-m-all (production default C-order kernels).

| Workload | Sites | C (s) | Julia (s) | Rust (s) | Julia/C | Rust/C |
|---|---:|---:|---:|---:|---:|---:|
| Opt | 32 | 12.163540 | 18.114558 | 8.467889 | 1.489 | 0.696 |
| Opt | 64 | 32.885330 | 60.391570 | 32.349213 | 1.836 | 0.984 |
| PhysCal | 32 | 4.013840 | 4.549924 | 3.459114 | 1.134 | 0.862 |
| PhysCal | 64 | 13.616030 | 22.451876 | 12.384418 | 1.649 | 0.910 |

C: fresh-process rank-zero internal All timer, after MPI_Init.
Julia/Rust: warmed production API maximum over ranks, including parsing,
initialization/parameter loading, sampling, optimization or observables, and output.
Process startup, JIT, build/setup and warmups are excluded. Timing boundaries differ.
Raw measurements, outputs, input hashes, source hashes, versions and linked libraries
are retained beside this report. Timing runs are sequential. Energies are not parity proof.
