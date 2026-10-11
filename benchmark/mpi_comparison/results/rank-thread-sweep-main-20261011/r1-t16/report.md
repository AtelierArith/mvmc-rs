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
| Opt | 32 | 11.376330 | 11.686872 | 6.901720 | 1.027 | 0.607 |
| Opt | 64 | 31.041090 | 36.263298 | 27.758737 | 1.168 | 0.894 |
| PhysCal | 32 | 3.909120 | 4.539372 | 2.584190 | 1.161 | 0.661 |
| PhysCal | 64 | 11.838060 | 16.776836 | 10.372454 | 1.417 | 0.876 |

C: fresh-process rank-zero internal All timer, after MPI_Init.
Julia/Rust: warmed production API maximum over ranks, including parsing,
initialization/parameter loading, sampling, optimization or observables, and output.
Process startup, JIT, build/setup and warmups are excluded. Timing boundaries differ.
Raw measurements, outputs, input hashes, source hashes, versions and linked libraries
are retained beside this report. Timing runs are sequential. Energies are not parity proof.
